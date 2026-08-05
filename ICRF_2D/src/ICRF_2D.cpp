// ============================================================================
//  ICRF_2D.cpp
//
//  ASGarD solver for the 2D BOUNCE-AVERAGED ICRF FOKKER-PLANCK equation
//  (Taosif Ahsan notes, Chapter 4, eqs 162-186).
//
//      M dF/dtau = sin(th0) d/dx0 (G~0)  +  d/dth0 (H~0)
//      G~0 = A~0 F + B~0 dF/dx0 + C~0 dF/dth0
//      H~0 = D~0 F + E~0 dF/dx0 + F~0 dF/dth0
//
//      For the full pitch domain, the signed cos(th0) mass is regularized in
//      the global ASGarD mass as |M| = x0^2 sin(th0) lambda(th0),
//
//  TABLES: loaded from tables/*.bin ONCE in main() into a plain Tables struct
//  (raw arrays).  All interpolation is done by the free functions below --
//  O(1) LINEAR interpolation over (v, pitch); psi is fixed at build time.
//  No file I/O ever happens during the solve.
// ============================================================================

#include "asgard.hpp"
#include "GL.hpp"

// The user-editable initial condition f0(x, theta, coll_eq): lives in
// input_data/ because it is an input, even though it is code.  See the
// header itself for the contract.
#include "initial_condition.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>

using P = asgard::default_precision;
static const P gauss_fact = 2.0/std::sqrt(M_PI);

double constexpr PI = asgard::PI;

// ============================================================================
//  TABLES : plain arrays, loaded once, passed around by pointer.
//
//     L, I                  : 1D (pitch)      L_tab.bin, I_tab.bin
//     res                   : 2D (x, pitch)   resonance geometry sum
//  row-major: field[ix*Npa + ipa]
struct Tables {
    std::vector<double> L, I;                    // 1D (pitch)
    std::vector<double> res;                     // 2D (x, pitch), without eps_E
    int Nx = 0, Npa = 0;
    double x0 = 0, xmax = 0, inv_hx = 0;
    double pa0 = 0, pamax = 0, inv_hpa = 0;
    double omega = 0, Omega_s0 = 0;
    double z_a = 0, m_a = 0, T_a = 0, logLambda_aa = 0;
};

static std::vector<double> load_bin(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    if (bytes < 0 || bytes % (long)sizeof(double) != 0) {
        std::fclose(f);
        throw std::runtime_error("Corrupt/!=8-byte-aligned file: " + path);
    }
    long n = bytes / (long)sizeof(double);
    std::rewind(f);
    std::vector<double> v(n);
    size_t got = std::fread(v.data(), sizeof(double), (size_t)n, f);
    std::fclose(f);
    if ((long)got != n)
        throw std::runtime_error("Short read in: " + path);
    return v;
}

// load all tables ONCE; called from main() before the solver starts
static Tables load_tables(const std::string& dir)
{
    std::string d = dir;
    if (!d.empty() && d.back() != '/') d += '/';

    Tables T;
    std::vector<double> parameters = load_bin(d + "parameters.bin");
    if (parameters.size() != 12)
        throw std::runtime_error("Tables: parameters.bin must contain 12 doubles; regenerate tables");

    T.L  = load_bin(d + "L_tab.bin");
    T.I  = load_bin(d + "I_tab.bin");
    T.res = load_bin(d + "res_tab.bin");

    T.Nx = static_cast<int>(std::llround(parameters[0]));
    T.Npa = static_cast<int>(std::llround(parameters[1]));
    T.x0 = parameters[2];
    T.xmax = parameters[3];
    T.pa0 = parameters[4];
    T.pamax = parameters[5];
    T.omega = parameters[6];
    T.Omega_s0 = parameters[7];
    T.z_a = parameters[8];
    T.m_a = parameters[9];
    T.T_a = parameters[10];
    T.logLambda_aa = parameters[11];

    if (T.Nx < 2 || T.Npa < 2)
        throw std::runtime_error("parameters.bin: tab_Nx and tab_Npitch must be >= 2");
    if (!(T.xmax > T.x0) || !(T.pamax > T.pa0))
        throw std::runtime_error("parameters.bin: table ranges must be increasing");

    auto chk = [](const char* nm, size_t got, size_t want) {
        if (got != want) throw std::runtime_error(
            std::string("Tables: ") + nm + " has " + std::to_string(got)
            + " values but grid implies " + std::to_string(want)
            + "  (regenerate: rm tables/*.bin && build_tables)");
    };
    chk("L_tab",  T.L.size(),  (size_t)T.Npa);
    chk("I_tab",  T.I.size(),  (size_t)T.Npa);
    chk("res_tab", T.res.size(), (size_t)T.Nx * T.Npa);

    T.inv_hx = (T.Nx - 1) / (T.xmax - T.x0);
    T.inv_hpa = (T.Npa - 1) / (T.pamax - T.pa0);
    return T;
}

// ============================================================================
//  LINEAR interpolation (free functions; T passed by pointer)
// ============================================================================

// O(1) uniform-grid cell locator: cell index i, weight t in [0,1]
static inline int locate(double g0, double inv_h, int n, double x, double& t)
{
    double pos = (x - g0) * inv_h;
    int i = (int)std::floor(pos);
    if (i < 0)     { t = 0.0; return 0;     }
    if (i > n - 2) { t = 1.0; return n - 2; }
    t = pos - (double)i;
    return i;
}

static inline double interp_1d_field(std::vector<double> const& field,
                                     Tables const* T, double pitch)
{
    double t;
    int i = locate(T->pa0, T->inv_hpa, T->Npa, pitch, t);
    return field[i] + (field[i + 1] - field[i]) * t;
}

static inline double interp_2d_field(std::vector<double> const& field,
                                     Tables const* T, double x, double pitch)
{
    double tv, tpa;
    int ix  = locate(T->x0,  T->inv_hx,  T->Nx,  x,     tv);
    int ipa = locate(T->pa0, T->inv_hpa, T->Npa, pitch, tpa);

    size_t k00 = (size_t)ix * T->Npa + ipa;
    double f00 = field[k00],             f10 = field[k00 + T->Npa];
    double f01 = field[k00 + 1],         f11 = field[k00 + T->Npa + 1];
    double a0 = f00 + (f10 - f00) * tv;
    double a1 = f01 + (f11 - f01) * tv;
    return a0 + (a1 - a0) * tpa;
}

// ============================================================================
//  helpers
// ============================================================================

// "1.0 2.0 3.0" -> vector<P>
static std::vector<P> str_to_vec(const std::string& input) {
    std::istringstream iss(input);
    std::vector<P> result;
    P value;
    while (iss >> value) result.push_back(value);
    return result;
}

// Psi(x) := (erf(x) - x erf'(x))/(2x^2), regularized near 0
static P Psi(P x){
    if (std::abs(x) < 1e-3){
        return x * gauss_fact * (1/3.0 - std::pow(x,2)/5.0);
    } else {
        P derf = gauss_fact * std::exp(-x*x);
        return (std::erf(x) - x * derf)/(2*std::pow(x,2));
    }
}

// ============================================================================
//  Coefficients : collisional (analytic) + quasilinear (table lookups).
//  Holds a raw pointer to the Tables owned by main() -- loaded once, read-only.
// ============================================================================

// Input units: charge in e, mass in amu, temperature in keV.
// The background n_bg input is a composition ratio; it is normalized to
// sum(n_bg)=1 here, so all densities below are already divided by n_ref.
// Velocity is normalized to the solved minority species a.  Time/collision
// strength is normalized with the table reference density used by build_tables:
//   C_b = Gamma_ab/Gamma_0, mu_b = m_b/m_a, l_b = v_ta/v_tb.
class Coefficients{
public:
    P omega, Omega_s0;
    P eps_E, cut_center, smoothing_width, x_domain_max;
    Tables const* tab;    // non-owning; main()'s Tables outlives the solver
    std::vector<P> C, l, mu;
    
    Coefficients(const std::vector<P>& z_bg_,
                 const std::vector<P>& m_bg_,
                 const std::vector<P>& n_bg_,
                 const std::vector<P>& T_bg_,
                 const std::vector<P>& logLambda_bg_,
                 const P T_e_,
                 const P logLambda_ea_,
                 const P eps_E_,
                 const P cut_center_,
                 const P smoothing_width_,
                 const P x_domain_max_,
                 Tables const* tab_):
    omega(static_cast<P>(tab_->omega)),
    Omega_s0(static_cast<P>(tab_->Omega_s0)),
    eps_E(eps_E_),
    cut_center(cut_center_),
    smoothing_width(smoothing_width_),
    x_domain_max(x_domain_max_),
    tab(tab_)
{
        if (!(x_domain_max > P{0}))
            throw std::runtime_error("x_domain_max must be positive");
        if (!(cut_center > P{0}))
            throw std::runtime_error("cut_center must be positive");
        if (smoothing_width < P{0})
            throw std::runtime_error("smoothing_width must be nonnegative");

        const std::size_t N = n_bg_.size();
        if (z_bg_.size() != N || m_bg_.size() != N || T_bg_.size() != N ||
            logLambda_bg_.size() != N)
            throw std::runtime_error("background ion arrays must have the same length");

        P n_sum = P{0};
        for(std::size_t i = 0; i < N; i++)
            n_sum += n_bg_[i];

        std::vector<P> n_frac(N);
        for(std::size_t i = 0; i < N; i++)
            n_frac[i] = n_bg_[i] / n_sum;

        P n_e = P{0};
        for(std::size_t i = 0; i < N; i++)
            n_e += n_frac[i] * z_bg_[i];

        const P m_e = 5.4461702149e-4; // electron mass normalized by amu
        const P z_a = static_cast<P>(tab_->z_a);
        const P m_a = static_cast<P>(tab_->m_a);
        const P T_a = static_cast<P>(tab_->T_a);
        const P logLambda_aa = static_cast<P>(tab_->logLambda_aa);
        const P denom = z_a * z_a * logLambda_aa;
        
        C.resize(N + 1); // Gamma_ab/Gamma_0
        l.resize(N + 1); // inverse thermal velocity
        mu.resize(N + 1); // mass ratio m_b/m_a
 
        // 0th entry is the electron: Gamma_ae/Gamma_0, with its own Coulomb log.
        C[0] = n_e * logLambda_ea_ / denom;
        mu[0] = m_e / m_a;
        l[0] = std::sqrt(mu[0] * T_a / T_e_);
        
        // 1..N are background/majority ions.
        for (size_t i = 0; i < N; i++){
            C[i + 1] = n_frac[i] * pow(z_bg_[i], 2) * logLambda_bg_[i] / denom;
            mu[i + 1] = m_bg_[i] / m_a;
            l[i + 1] = std::sqrt(mu[i + 1] * T_a / T_bg_[i]);
        }
    }
    
    // ---- collisional v-parts (analytic) ------------------------------------
    P A_cv(P x) const{
        P sum = 0;
        for (size_t b = 0; b < C.size(); ++b){
            P px = Psi(l[b] * x);
            sum += (2.0 * C[b] * l[b] * l[b] / mu[b]) * x * x * px;
        }
        return sum;
    }
    P B_cv(P x) const{
        P sum = 0;
        for (size_t b = 0; b < C.size(); ++b) {
            P px = Psi(l[b] * x);
            sum += C[b] * x * px;
        }
        return sum;
    }
    
    P F_cv(P x) const{
        if(x<1e-3){
            P sum = 0;
            for (size_t b = 0; b < C.size(); ++b)
                sum += C[b] * l[b] * 2/(3*std::sqrt(M_PI));
            return sum;
        } else {
            P sum = 0;
            for (size_t b = 0; b < C.size(); ++b) {
                P px = Psi(l[b] * x);
                sum += C[b] * (std::erf(l[b] * x) - px) / (2.0 * x);
            }
            return sum;
        }
    }
    
    P collisional_equilibrium(P x) const{
        if (x <= P{0.0}) return P{1.0};
        P const integral = gl::integrate<16>([this](P u) {
            P const b = B_cv(u);
            return (std::abs(b) > std::numeric_limits<P>::min())
                 ? A_cv(u) / b : P{0.0};
        }, P{0.0}, x);
        return std::exp(-integral);
    }


    // ---- table lookups (LINEAR, O(1)) --------------------------------------
    P L(P th) const{ return interp_1d_field(tab->L, tab, th); }
    P I(P th) const{ return interp_1d_field(tab->I, tab, th); }

    P lambda(P th) const{
        return std::abs(std::cos(th)) * L(th);
    }

    // Separable factors of the evolution mass
    // M(x,theta)=mass_x(x)*mass_th(theta).
    P mass_x(P x) const {
        return x * x;
    }

    P mass_th(P th) const {
        return std::sin(th) * lambda(th);
    }

    P resonance(P x, P th) const {
        if (x < P{1e-30} || !(omega > P{0}))
            return P{0};
        return interp_2d_field(tab->res, tab, x, th);
    }
    
    // Normalization integral for the equilibrium velocity distribution.
    // GL64 is ample for this smooth finite-domain integral and replaces the
    // former 4097-point trapezoid loop.
    P collisional_x_norm(P x_max) const{
        return gl::integrate<64>([this](P x) {
            return mass_x(x) * collisional_equilibrium(x);
        }, P{0.0}, x_max);
    }
    
    // General form: integral of mass_th(theta) * g(theta) over 0..pi, with
    // the same trapped-passing-peak substitution as the plain norm -- the
    // peak lives in mass_th, so any smooth extra shape g rides along
    // accurately.  Used by the initial-condition normalization, where g is
    // the user f0's pitch dependence at one x.
    template<typename G>
    P collisional_th_norm_shaped(G g) const{
        // L has a narrow peak at the trapped-passing boundary.  Locate that
        // boundary from the first-half pitch table, transform the distance from
        // that boundary on each side, and use pitch symmetry for the other half.
        // This prevents one global rule from stepping over the peak.
        int const half_end = std::min(
            tab->Npa - 1,
            static_cast<int>(std::floor((P{0.5} * PI - tab->pa0)
                                      * tab->inv_hpa)));
        int turn_index = 0;
        for (int i = 1; i <= half_end; ++i)
            if (tab->L[i] > tab->L[turn_index]) turn_index = i;

        P const theta_turn = static_cast<P>(
            tab->pa0 + static_cast<double>(turn_index) / tab->inv_hpa);
        if (!(theta_turn > P{0}) || !(theta_turn < P{0.5} * PI))
            throw std::runtime_error(
                "could not locate trapped-passing peak in L table");

        // Resolve the narrow peak without evaluating its endpoint directly.
        // On either side use |theta-theta_turn|=u^2, so uniformly distributed
        // GL nodes in u cluster quadratically close to the peak.  The factor
        // 2*u is dtheta/du.
        auto integrate_side = [this, theta_turn, g](P theta_span, P direction) {
            auto transformed = [this, theta_turn, direction, g](P u) {
                P const th = theta_turn + direction * u * u;
                return P{2} * u * mass_th(th) * g(th);
            };
            return gl::integrate<32>(
                transformed, P{0}, std::sqrt(theta_span));
        };

        return P{2} * (
            integrate_side(theta_turn, P{-1})
          + integrate_side(P{0.5} * PI - theta_turn, P{1}));
    }

    P collisional_th_norm() const{
        return collisional_th_norm_shaped([](P) { return P{1}; });
    }
    
    // A center at or beyond x_max explicitly disables the artificial RF
    // cutoff.  In that mode the physical QL coefficients extend to x_max.
    bool use_cutoff() const {
        return cut_center < P{1};
    }

    // Smoothly remove RF diffusion before the finite velocity boundary.  Both
    // input parameters are fractions of x_max: the multiplier is 1 in the
    // interior, 1/2 at cut_center*x_max, and approaches 0 over the
    // dimensionless width smoothing_width. cut_center >= 1 disables the
    // multiplier. With smoothing_width == 0, use a sharp Heaviside step.
    P cutoff(P x) const {
        if (!use_cutoff())
            return P{1};
        P const x_fraction = x / x_domain_max;
        if (smoothing_width == P{0})
            return x_fraction < cut_center ? P{1} : P{0};
        return P{0.5} * (P{1} - std::tanh(
            (x_fraction - cut_center) / smoothing_width));
    }

    // Named separable factors of the QL tensor. C and E share the same radial
    // and pitch factors, which makes both symmetry and B*F=C*E explicit.
    P ql_B_x(P x) const {
        return eps_E * cutoff(x) * x * x;
    }

    P ql_CE_x(P x) const {
        return eps_E * cutoff(x) * x;
    }

    P ql_F_x(P x) const {
        return eps_E * cutoff(x);
    }

    P ql_theta_base(P th) const {
        P const L_pitch = L(th);
        P const costh = std::cos(th);
        if (!(L_pitch > P{1e-30}) || std::abs(costh) < P{1e-4})
            return P{0};
        return lambda(th) / L_pitch;
    }

    P ql_B_theta(P th) const {
        P const sinth = std::sin(th);
        return ql_theta_base(th) * sinth * sinth * sinth;
    }

    P ql_CE_theta(P th) const {
        P const sinth = std::sin(th);
        P const costh = std::cos(th);
        return ql_theta_base(th) * sinth * sinth / costh
             * (Omega_s0 / omega - sinth * sinth);
    }

    P ql_F_theta(P th) const {
        P const sinth = std::sin(th);
        P const costh = std::cos(th);
        P const alpha_minus_sin2 = Omega_s0 / omega - sinth * sinth;
        return ql_theta_base(th) * sinth / (costh * costh)
             * alpha_minus_sin2 * alpha_minus_sin2;
    }

};

// ============================================================================
//  PDE construction
// ======================================= =====================================
asgard::pde_scheme<P> make_icrf_pde(asgard::prog_opts options, Tables const* tab)
{
    using term_volume = asgard::term_volume<P>;
    using term_div    = asgard::term_div<P>;
    using term_grad   = asgard::term_grad<P>;
    using term_1d     = asgard::term_1d<P>;
    using term_md     = asgard::term_md<P>;
    using flux        = asgard::flux_type;
    using bc          = asgard::boundary_type;

    using term_identity = asgard::term_identity;
    using term_interp   = asgard::term_interp<P>;

    // -- velocity domain -----------------------------------------------------
    constexpr P x_min = P{0.0};
    const P x_max = options.file_required<P>("x_max");
    if (!(x_max > x_min)) {
        throw std::runtime_error("velocity domain needs x_max > 0");
    }
    // -- physics parameters --------------------------------------------------
    Coefficients cf(str_to_vec(options.file_required<std::string>("z_bg")),
                    str_to_vec(options.file_required<std::string>("m_bg")),
                    str_to_vec(options.file_required<std::string>("n_bg")),
                    str_to_vec(options.file_required<std::string>("T_bg")),
                    str_to_vec(options.file_required<std::string>("logLambda_bg")),
                    options.file_required<P>("T_e"),
                    options.file_required<P>("logLambda_ea"),
                    options.file_required<P>("eps_E"),
                    options.file_value<P>("cut_center").value_or(P{1.0}),
                    options.file_value<P>("smoothing_width").value_or(P{0.02}),
                    x_max,
                    tab);

    bool const use_ql = (cf.eps_E != P{0.0});
    if (use_ql &&
        x_max > static_cast<P>(tab->xmax) + 100 * std::numeric_limits<P>::epsilon()) {
        throw std::runtime_error("x_max exceeds the QL table x range; "
                                 "regenerate tables with tab_xmax >= x_max");
    }

    options.title = "ICRF 2D bounce-averaged Fokker-Planck";
    P th_min = 0.0;
    P th_max = PI;
    
    asgard::pde_domain<P> domain({{x_min, x_max}, {th_min, th_max}});
    domain.set_names({"x0", "theta"});

    // -- discretization options ----------------------------------------------
    options.default_degree = 2;
    options.default_start_levels = {5, };

    options.default_step_method = asgard::time_method::cn;

    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::jacobi;

    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 50;

    asgard::pde_scheme<P> pde(options, std::move(domain));

    P const theta_norm = cf.collisional_th_norm();
    P const x_norm = cf.collisional_x_norm(x_max);
    if (!(x_norm > P{0.0}) || !(theta_norm > P{0.0}))
        throw std::runtime_error("velocity normalization is non-positive");

    // ========================================================================
    //  MASS  M = mass_x(x0) * mass_th(theta)
    //        = x0^2 * sin(theta) * cos_eps(theta) * L~
    // ========================================================================
    
    
    auto mass_x = [cf](std::vector<P> const& x, std::vector<P>& value) {
        for (std::size_t i = 0; i < x.size(); ++i)
            value[i] = cf.mass_x(x[i]);
    };
    auto mass_th = [cf](std::vector<P> const& th, std::vector<P>& value) {
        for (std::size_t i = 0; i < th.size(); ++i)
            value[i] = cf.mass_th(th[i]);
    };
    
    pde.set_mass({term_volume{mass_x}, term_volume{mass_th}});

    // ========================================================================
    //  COLLISIONAL TERMS  (separable)
    // ========================================================================
    
    // (C1) collisional A : x0-advection
    {
        auto neg_A_cv = [cf](std::vector<P> const& x,
                             std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = -cf.A_cv(x[i]);
        };
        pde += term_md({term_div(neg_A_cv, flux::upwind),
                        term_identity{}});
    }

    // (C2) collisional B : x0-diffusion, collisional zero-flux Robin + penalty
    {
        auto neg_B_cv = [cf](std::vector<P> const& x,
                             std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = -cf.B_cv(x[i]);
        };
        
        // Homogeneous Neumann (f_x=0) at both radial boundaries.
        term_1d div_grad_xc({
            term_div(neg_B_cv, flux::upwind, bc::bothsides),
            term_grad(mass_x),
        });
        // ------------------------------------------------------------------
        // Zero TOTAL flux at the outer wall.
        //
        // Steady state is zero FLUX, not zero gradient: with
        //     Gamma_x = (B_cv + D^QL_xx) df/dx + D^QL_xth df/dth + A_cv f,
        // homogeneous Neumann (df/dx = 0) leaves the drag and quasilinear
        // fluxes uncancelled at the boundary.  Measured: that MANUFACTURES
        // particles once the tail reaches the wall -- +63% by t = 100
        // (level 5, x_max = 6), still accelerating, and worse under
        // refinement.  The condition we want is Gamma_x = Gamma_theta = 0.
        //
        // Those look coupled (Gamma_x contains df/dtheta), but the QL tensor
        // is rank-1 -- B0*F0 = C0*E0, enforced by construction below -- so
        // zeroing BOTH components forces v.grad(f) = 0 and the entire QL
        // flux drops out.  What survives is purely collisional,
        //     B_cv df/dx + A_cv f = 0,
        // a plain Robin with no theta dependence.  Hence bc::bothsides on
        // the QL divergences (see the QL block) plus this Robin.
        //
        // On the VALUE: asgard's Robin adds a boundary FLUX, so it takes the
        // mass-weighted drag coefficient -- the same quantity handed to the
        // drag term as neg_A_cv -- not the log-derivative A_cv/B_cv.  A_cv
        // is already mass-weighted: A_cv/x^2 -> 1/x^2 and B_cv/x^2 -> 1/x^3
        // at large x, exactly the physical coefficients (compare LHCD_2D,
        // whose drag literal 0.5 is x^2 * 1/(2x^2), constant only because
        // its A is a pure power).  Getting this wrong is not subtle:
        // A_cv/B_cv = 2*x_max overshoots ~14x and the solve diverges;
        // A_cv/x^2 undershoots 36x and drifts to +5% where A_cv gives -0.35%.
        //
        // The left end needs nothing: A_cv ~ x^2 Psi(x) -> 0 as x -> 0, so
        // the flux vanishes there on its own.
        // ------------------------------------------------------------------
        div_grad_xc.set_left_robin(P{0.0});
        div_grad_xc.set_right_robin(cf.A_cv(x_max));
        
        pde += term_md({div_grad_xc, term_identity{}});

        P const inv_dx = 1.0 / pde.cell_size(asgard::dimension_id{0});
        pde += term_md({
            asgard::term_penalty<P>{inv_dx, bc::none},
            term_identity{},
        });
    }

    // (C3) collisional F : theta-diffusion + penalty
    {
        auto neg_I_sinth = [cf](std::vector<P> const& th,
                                std::vector<P>& value) {
            for (std::size_t i = 0; i < th.size(); ++i)
                value[i] = -cf.I(th[i]) * std::sin(th[i]);
        };
        auto L_sinth = [cf](std::vector<P> const& th,
                            std::vector<P>& value) {
            for (std::size_t i = 0; i < th.size(); ++i)
                value[i] = cf.L(th[i]) * std::sin(th[i]);
        };
        auto F_cv = [cf](std::vector<P> const& x,
                         std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = cf.F_cv(x[i]);
        };
        
        // The pitch mass uses cos_eps instead of signed cos(theta).  This keeps
        // the mass positive through theta=pi/2; the branch sign is not part of
        // this collisional pitch-diffusion term.

        term_1d div_grad_thc({
            term_div(neg_I_sinth, flux::upwind, bc::bothsides),
            term_grad(L_sinth)});

        // u = 1/(x^2 L cos_eps sinth) d/dth(pitch-diffusion flux)
        pde += term_md({term_volume{F_cv}, div_grad_thc});

        P const inv_dth = 1.0 / pde.cell_size(asgard::dimension_id{1});
        
        pde += term_md({
            term_identity{},
            asgard::term_penalty<P>{inv_dth, bc::none}
        });
    }

    // ========================================================================
    //  QUASILINEAR TERMS  (non-separable -> interpolation)
    // ========================================================================
    if (use_ql) {
        // Interpolate only the genuinely non-separable resonance amplitude.
        // The remaining B,C,E,F geometry is separable and is absorbed into
        // the corresponding gradient below.  All four tensor entries then use
        // the same interpolated R instead of four independent projections,
        // which do not preserve the nonlinear identity B*F=C*E.
        // These operators act on the conservative density
        // x^2*sin(theta)*lambda*f.
        // bc::bothsides closes the QL flux at the domain edges, and is
        // required TOGETHER with the collisional Robin above: at the wall the
        // QL flux dominates (D^QL_xx ~ 17 vs B_cv ~ 0.1, and cut_center >= 1
        // leaves the RF live all the way out), so leaving these as bc::none
        // (outflow) bleeds particles however good the collisional condition
        // is -- measured -11.6% by t = 200 with the Robin alone, versus
        // -0.4% with both.  At x = 0 and theta = 0, pi the QL flux vanishes
        // geometrically, so only x_max binds.
        term_md div_dx({term_div{P{-1.0}, flux::upwind, bc::bothsides},
                        term_volume{P{1.0}}});
        term_md div_theta({term_volume{P{1.0}},
                           term_div{P{-1.0}, flux::upwind, bc::bothsides}});
        auto R_ql = [cf](P, asgard::vector2d<P> const& nodes,
                         std::vector<P> const& input,
                         std::vector<P>& output) {
            for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
                P const x = nodes[i][0];
                P const th = nodes[i][1];
                output[i] = cf.resonance(x, th) * input[i];
            }
        };

        // With sin(theta) already absorbed into B and C, write
        //
        //   [B C; E F] = res(x,theta) [B0 C0; E0 F0],
        //
        // where every K_ij is separable and B0*F0=C0*E0 exactly.
        auto B_x = [cf](std::vector<P> const& x, std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = cf.ql_B_x(x[i]);
        };
        auto CE_x = [cf](std::vector<P> const& x, std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = cf.ql_CE_x(x[i]);
        };
        auto F_x = [cf](std::vector<P> const& x, std::vector<P>& value) {
            for (std::size_t i = 0; i < x.size(); ++i)
                value[i] = cf.ql_F_x(x[i]);
        };
        auto B_theta = [cf](std::vector<P> const& th,
                            std::vector<P>& value) {
            for (std::size_t i = 0; i < th.size(); ++i)
                value[i] = cf.ql_B_theta(th[i]);
        };
        auto CE_theta = [cf](std::vector<P> const& th,
                             std::vector<P>& value) {
            for (std::size_t i = 0; i < th.size(); ++i)
                value[i] = cf.ql_CE_theta(th[i]);
        };
        auto F_theta = [cf](std::vector<P> const& th,
                            std::vector<P>& value) {
            for (std::size_t i = 0; i < th.size(); ++i)
                value[i] = cf.ql_F_theta(th[i]);
        };

        // Put each separable K_ij directly inside Grad_j.  This is the
        // multidimensional Grad[q] form verified by the mass-chain probe and
        // avoids an additional intermediate projection/mass operation.
        term_md grad_B({term_grad{B_x}, term_volume{B_theta}});
        term_md grad_C({term_volume{CE_x}, term_grad{CE_theta}});
        term_md grad_E({term_grad{CE_x}, term_volume{CE_theta}});
        term_md grad_F({term_volume{F_x}, term_grad{F_theta}});

        term_md res = term_interp{R_ql};

        pde += term_md({div_dx,    res, grad_B});
        pde += term_md({div_dx,    res, grad_C});
        pde += term_md({div_theta, res, grad_E});
        pde += term_md({div_theta, res, grad_F});
        
        // The only non-separable quantity interpolated by the QL operator is
        // the resonance amplitude, so use that same interpolation as the
        // adaptive-grid weight.
        auto weight_ql = [cf](P, asgard::vector2d<P> const& nodes,
                              std::vector<P> const& input,
                              std::vector<P>& output) {
            for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
                P const x = nodes[i][0];
                P const th = nodes[i][1];
                output[i] = std::abs(cf.resonance(x, th))
                          * std::abs(input[i]);
            }
        };
        pde.set_adapt_weight(weight_ql);
    }

    // ========================================================================
    //  INITIAL CONDITION : f0(x, theta, coll_eq) from
    //  input_data/initial_condition.hpp (a compile input: editing it triggers
    //  a rebuild and a re-solve).  Fully 2-D; the default returns the
    //  zero-flux collisional equilibrium, reproducing the old behavior.
    // ========================================================================
    // Normalize against the evolution measure x^2 lambda(theta) sin(theta),
    // with the gyrophase 2 pi folded in: the full 3-D particle count starts
    // at exactly 1 (the solver's conserved 2-D weighted moment therefore at
    // 1/(2 pi)), matching LHCD_2D's convention so the density_* plots of
    // both projects read N = 1.  The x integral (GL64, smooth) nests the
    // peak-split theta quadrature so a pitch-dependent f0 still integrates
    // accurately across the trapped-passing boundary.
    P const init_norm = P{2} * PI * gl::integrate<64>([cf](P x) {
        P const eq = cf.collisional_equilibrium(x);
        return cf.mass_x(x) * cf.collisional_th_norm_shaped(
            [x, eq](P th) { return static_cast<P>(initial_f0(x, th, eq)); });
    }, P{0.0}, x_max);
    if (!(init_norm > P{0.0}) || !std::isfinite(init_norm))
        throw std::runtime_error(
            "initial_condition.hpp: integral of f0 against the evolution "
            "measure must be positive and finite");
    // Unlike the old separable path (which projected mass-weighted values
    // against the local mass matrices), the non-separable initial condition
    // goes through asgard's interpolation machinery, which collocates the
    // function *itself* -- so supply plain f0, no evolution measure.
    pde.set_initial([cf, init_norm](P, asgard::vector2d<P> const &nodes,
                                    std::vector<P> &fx) {
        for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
            P const x = nodes[i][0];
            P const th = nodes[i][1];
            fx[i] = initial_f0(x, th, cf.collisional_equilibrium(x))
                  / init_norm;
        }
    });

    return pde;
}

// ============================================================================
//  main()
// ============================================================================
int main(int argc, char** argv)
{
    asgard::libasgard_runtime running_(argc, argv);
    asgard::prog_opts options(argc, argv);

    if (options.show_help) {
        std::cout <<
          "\n 2D bounce-averaged ICRF Fokker-Planck (PDF eqs 162-186)\n"
          "   dim0 = x0 (=v/vta),  dim1 = theta0 (pitch, rad)\n"
          "   mass/Jacobian = x0^2 sin(theta0) lambda(theta0)\n"
          " Run with the ASGarD input deck:\n"
          "     ./ICRF_2D -if input_solver.txt [asgard opts]\n"
          " tables/ must already exist (build_tables).\n\n"
          "    -- standard ASGarD options --";
        options.print_help(std::cout);
        return 0;
    }

    // load the tables ONCE; everything downstream reads this by pointer.
    // Declared before `disc`, so it is destroyed after it -- the pointer held
    // by the lambdas remains valid for the whole solve.
    Tables tables = load_tables("tables/");

    asgard::discretization_manager<P> disc(make_icrf_pde(options, &tables),
        asgard::verbosity_level::high);

    int movie_stride = options.file_value<int>("movie_stride").value_or(0);
    int const movie_frames = options.file_value<int>("movie_frames").value_or(0);
    std::string const movie_dir = options.file_value<std::string>("movie_dir").value_or("movie");

    if (movie_stride <= 0 && movie_frames > 1) {
        int64_t const total_steps = disc.remaining_steps();
        movie_stride = static_cast<int>(
            std::max<int64_t>(1, (total_steps + movie_frames - 2) / (movie_frames - 1)));
    }

    bool success = true;
    if (movie_stride > 0) {
        std::filesystem::path const movie_path(movie_dir);
        std::filesystem::create_directories(movie_path);
        for (auto const &entry : std::filesystem::directory_iterator(movie_path)) {
            if (!entry.is_regular_file()) continue;
            std::string const filename = entry.path().filename().string();
            std::string const ext = entry.path().extension().string();
            if (filename.rfind("snapshot_", 0) == 0 && (ext == ".h5" || ext == ".hdf5"))
                std::filesystem::remove(entry.path());
        }

        int frame_index = 0;
        auto save_movie_snapshot = [&]() {
            std::ostringstream name;
            name << "snapshot_"
                 << std::setw(6) << std::setfill('0') << frame_index++
                 << "_step_"
                 << std::setw(8) << std::setfill('0') << disc.current_step()
                 << ".h5";
            disc.save_snapshot(movie_path / name.str());
        };

        save_movie_snapshot();
        while (disc.remaining_steps() > 0) {
            int64_t const chunk = std::min<int64_t>(movie_stride, disc.remaining_steps());
            success = disc.advance_time(chunk);
            save_movie_snapshot();
            if (!success) break;
        }
    } else {
        success = disc.advance_time();
    }

    disc.final_output();

    if (!success || disc.remaining_steps() > 0) {
        std::cerr << "ICRF_2D: time advancement stopped before the requested "
                     "final time.\n";
        return 2;
    }

    return 0;
}

