// ────────────────────────────────────────────────────────────────────────────────────────────
// Description
// ────────────────────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────────────────────
// We are solving the Stix homogenous equation with the following normalization scales
// mass, m_0 = 1 amu
// charge number, z_0 = 1
// temperature, T_0 = Electron Temperature
// density, n_0 = Electron Density
// inverse velocity, l = sqrt(m_0/2k_BT_0)
// Collision coefficient, C = 8pi n_0 e^4 ln Lambda/m_0^2
// inverse time, nu = C l^3)
// Dimensionless velocity is scaled by v -> lv
// Dimensionless time is scaled by t -> nu t
// l and nu can be thought of as ~ typical ion scales because T_i ~ T_e and m_i ~ m_p
// ────────────────────────────────────────────────────────────────────────────────────────────



// ────────────────────────────────────────────────────────────────────────────────────────────
// Section 0 : Aliases, Constants, Libraries, Necessary Static Function
// ────────────────────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────────────────────
// Included standard c++ library and the ASGarD library
// Defined constants such as pi, norm, electron charge and mass
// Created convenient aliases for common C++ and ASGarD types
// ────────────────────────────────────────────────────────────────────────────────────────────
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include "asgard.hpp"

// Use ASGarD's default precision (typically double)
using P = asgard::default_precision;

// Define Constants
const P pi = 3.14159265358979323846; // Value of Pi
const P gauss_fact = 2/sqrt(pi); //norm of gaussian

// Convenient aliases for common C++ types
using vector = std::vector<P>;
using string = std::string;

// Convenient aliases for common ASGarD types
using term_1d = asgard::term_1d<P>;
using term_md = asgard::term_md<P>;
using term_div = asgard::term_div<P>;
using term_volume = asgard::term_volume<P>;
using term_grad = asgard::term_grad<P>;
using separable_func = asgard::separable_func<P>;
using asgard::boundary_type;
using asgard::flux_type;

// ────────────────────────────────────────────────────────────────────────────────────────────
// Section 1 : Static methods for necessary functions and conversion
// ────────────────────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────────────────────
// Defined string to vector method
// Defined G_x(x) := G(x)/x function where, G(x) := (erf(x) - x erf'(x))/(2x^2).
// Defined it first to regularize the value of G(x) near x = 0 better in code
// Defined G(x) function retrieved from G(x) = x G_x(x)
// ────────────────────────────────────────────────────────────────────────────────────────────

// Method to convert string format to vector data format
static vector str_to_vec(const string& input) {
    std::istringstream iss(input);
    vector result;
    P value;
    while (iss >> value) {
        result.push_back(value);
    }
    return result;
}

// Return G(x) := (erf(x) - x erf'(x))/(2x^2) function
// returns G(x)/x function if divide is set to true
static P G(P x, bool divide){
    // Regularizes G(x)/x value near x = 0 via fourth order Taylor Expansion
    // Error is below 1E-20 for x < 1e-3, under machine error
    P output;
    if (std::abs(x) < 1e-3){
        output = gauss_fact * (1/3.0 - pow(x,2)/5.0);
    }
    // Returns full expression otherwise
    else{
        P derf = gauss_fact * std::exp(-x*x); // The gaussian function, also phi'(x)
        output = (std::erf(x) - x * derf)/(2*pow(x,3)); // Returns G_x = G(x)/x
    }
    // returns G(x)/x if divide, returns G(x)
    return divide? output : x * output;
}

// ────────────────────────────────────────────────────────────────────────────────────────────
// Section 2 : Class Object for the plasma data comprised of various species
// ────────────────────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────────────────────
// Specifies power, charges, masses, densities and temperatures of ions
// Specifies temperature of electrons. Charge, mass and density of electron is deduced.
// Allows choice of a test particle.
// Functions such as eta(v), zeta(v) are calculated
// eta(v) := - alpha(v) + 0.5 /v^2  d/dv(v^2 beta(v))
// zeta(v) := 0.5 beta(v) + K
// ────────────────────────────────────────────────────────────────────────────────────────────

class PlasmaData{
public:
    P K; // Electron temperature and plasma power (normalized)
    P z, m; // Minority charges, masses
    vector z_ion, m_ion, n_ion, T_ion;  // Ion charges, masses, densities and temperatures
    vector l, beta_coeff, eta_coeff; // All inverse thermal velocity and coefficients for beta, eta
    
    // Constructor
    PlasmaData(const P K_, const P z_, const P m_,
               const vector& z_ion_, const vector& m_ion_,  const vector& n_ion_, const vector& T_ion_):
            K(K_), z(z_), m(m_),
            z_ion(z_ion_), m_ion(m_ion_), n_ion(n_ion_), T_ion(T_ion_)
    {
        const int N = n_ion.size(); // Number of ion species
        // Rescaling the density of ions to match quasi neutrality
        
        // n_total = sum_i(n_i z_i)
        P n_total = 0;
        for(int i = 0; i < N; i++){
            n_total += n_ion[i] * z_ion[i];
        }
        
        // n_i -> n_i/n_total => n_e(-1) + sum_i n_i z_i = - 1 + 1 = 0
        for(int i = 0; i < N; i++){
            n_ion[i] = n_ion[i]/n_total;
        }

        
        // Electron parameters after normalization
        const P T_e = 1; // Electron Temperature normalized by itself
        const P z_e = - 1.0; // electron charge normalized by itself
        const P m_e = 5.4461702149e-4; // electron mass normalized by amu
        const P n_e = 1; // Electron density normalized by itself
        
        // Initializing the summation arrays
        P Cf; // Coulumb collisional coefficient
        l.resize(N + 1); // inverse thermal velocity
        beta_coeff.resize(N + 1); // beta coeffcients
        eta_coeff.resize(N + 1); // eta coeffcients
 
        // 0th entry in the arrays is the electron
        Cf = n_e * pow(z_e, 2) * pow(z, 2) / pow(m, 2);
        l[0] = sqrt(0.5 * m_e/T_e);
        eta_coeff[0] = Cf * pow(l[0], 2) * m / m_e;
        beta_coeff[0] = Cf * l[0];
        
        
        // 1 to nth entries in the arrays are the ions
        for (int i = 0; i < N; i++){
            Cf = n_ion[i] * pow(z_ion[i], 2) * pow(z, 2) / pow(m, 2);
            l[i + 1] = sqrt(0.5 * m_ion[i]/T_ion[i]);
            eta_coeff[i + 1] = Cf * pow(l[i + 1], 2) * (m / m_ion[i]);
            beta_coeff[i + 1] = Cf * l[i + 1];
        }
    }
    
    
    // eta(v) := - alpha(v) + 0.5 /v^2  d/dv(v^2 beta(v))
    // where, alpha(v) := <v_parallel> + 1/v <v_perpendicular^2>
    P eta(P v) const{
        // -alpha v^2 + 0.5 * d/dv (beta v^2) = sum_f C_f l_f^2 (m_test/m_f) G(l_f v)
        // Thus, eta(v) = sum_f eta_coeff_f G(l_f v, false)
        P eta_func = 0;
        for(int f = 0; f < l.size(); f++)
            eta_func += eta_coeff[f] * G(l[f] * v, false);
        return eta_func;
    }
    
    // zeta(v) := 0.5 beta(v) + K
    // beta(v) = <v_parallel^2> = sum_f C_f/v G(l_f v)
    P zeta(P v) const{
        // C_f G(l_f v)/v = C_f l_f G(l_f v)/(l_f v)
        // Thus, beta(v) = sum_f beta_coeff_f G(l_f v, true)
        P beta_func = 0;
        for(int f = 0; f < l.size(); f++)
            beta_func += beta_coeff[f] * G(l[f] * v, true);
        P zeta_func = 0.5 * beta_func + K;

        return zeta_func;
    }
};

// ────────────────────────────────────────────────────────────────────────────────────────────
// Section 3 : PDE Constuction
// ────────────────────────────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────────────────────────────
// PDE: df/dt -1/v^2 d/dv[ 1/2 beta(v) v^2 df/fv)] -1/v^2 d/dv[eta(v) v^2 f] = 0
// First term: -1/v^2 d/dv[ 1/2 beta(v) v^2 df/fv)]
// Second term: -1/v^2 d/dv[eta(v) v^2 f]
// Initial Condition: f(0,v) = 4/sqrt(pi) * exp(-v^2)
// Particle number : N := int_0^\infty f(v) v^2 dv = 1, conservation is ensured
// Boundary Condition: df/dv(t, infty) = 0, df/dv(t, infty) = 0
// Mass matrix is set to v^2 to avoid division by 0
// ────────────────────────────────────────────────────────────────────────────────────────────

asgard::pde_scheme<P> make_pde(asgard::prog_opts options)
{
    // Constructing plasma data object from input file for
    PlasmaData plasma(
                      options.file_required<P>("K"),
                      options.file_required<P>("z"),
                      options.file_required<P>("m"),
                      str_to_vec(options.file_required<string>("z_ion")),
                      str_to_vec(options.file_required<string>("m_ion")),
                      str_to_vec(options.file_required<string>("n_ion")),
                      str_to_vec(options.file_required<string>("T_ion"))
                      );
    
    // Default degree and level, unless set from input file
    options.default_degree = 1;
    options.default_start_levels = {8, };

    // Setting total run time and time step
    options.default_stop_time = options.file_required<P>("time");
    options.default_dt = options.file_required<P>("time_step");

    // Setting solver method for time steps
    options.default_step_method = asgard::time_method::cn; // Crank-Nicolson Method
    options.default_solver = asgard::solver_method::direct; // Direct solver
    options.default_precon = asgard::precon_method::jacobi; // Jacobi Precondiner
    
    P const v_min = options.file_required<P>("domain_min");
    P const v_max = options.file_required<P>("domain_max");
    
    // Setting domain of the PDE
    asgard::pde_domain<P> domain({{v_min, v_max}});
    
    // Declaring the PDE
    asgard::pde_scheme<P> pde(options, std::move(domain));
    
    // function form of v^2
    auto v2 = [](vector const &v, vector &f) -> void {
            for (size_t i = 0; i < v.size(); i++)
                f[i] = pow(v[i], 2);
    };
    
    // Jacobian mass matrix set to v^2
    pde.set_mass({term_volume(v2)});
    
    // q = - 1/v^2 d/dv[eta(v) v^2 f]
    {   // - eta(v) v^2
        auto eta_v2 = [plasma](vector const &v, vector &f)-> void {
            for (size_t i = 0; i < v.size(); i++){
                f[i] = - pow(v[i], 2) * plasma.eta(v[i]);
            }
        };
        pde += term_1d({term_div(eta_v2)}); // v^2 q = d/dv[ - v^2 eta(v) f]
    }
    
    // s = - 1/v^2 d/dv[ zeta(v) v^2 df/fv)]
    {
        // - zeta(v) v^2
        auto zeta_v2 = [plasma](vector const &v, vector &f) -> void {
            for (size_t i = 0; i < v.size(); i++)
                f[i] = -  pow(v[i], 2) * plasma.zeta(v[i]);
        };
        // asgard 0.9.1 chain form: a brace list of the chained 1-D terms,
        // matching the 2D solvers (the old explicit term_chain{} wrapper was
        // removed from the API).
        term_1d div_grad({
                term_div(zeta_v2, boundary_type::bothsides), // v^2 r = d/dv[ - v^2 zeta(v) s]
                term_grad(v2, boundary_type::none) // v^2 s = v^2 df/dv
        });
        // boundary conditions
        P const right =  pow(v_max, 2) * plasma.eta(v_max);
        div_grad.set_right_robin(right);

        P const inv_dv = 1.0/pde.cell_size(asgard::dimension_id{0});
        div_grad.set_penalty(inv_dv);
        
        pde += term_md({div_grad,});
    }
    
    // initial condition f(0,v) = f_0(v) := 4/sqrt(pi) * exp(-v^2)
    {
        P m_test = options.file_required<P>("m");
        // init := f_0(v) * v^2
        auto init = [m_test](vector const &v, P , vector &func){
            for (size_t i = 0; i < v.size(); i++){
                P f_0 =  2 * gauss_fact * std::exp(-m_test * pow(v[i],2)); //f_0(v)
                func[i] =  pow(m_test,1.5) * pow(v[i],2) * f_0; // v^2 f_0
            }
           
        };
        
        pde.add_initial(separable_func({init}));// v^2 f(v,0) = v^2 f_0(v)
    }
    
    // df/dt - 1/v^2 d/dv [v^2 (eta(v) f + zeta(v) df/fv)] = 0
    return pde;
}

// ────────────────────────────────────────────────────────────────────────────────────────────
// Section 4 : Simulation
// ────────────────────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    // Parse options from CLI or config file
    asgard::prog_opts options(argc, argv);

    // Construct PDE and discretization manager
    asgard::discretization_manager<P> disc(make_pde(options), asgard::verbosity_level::high);

    // Movie snapshots: same deck keys and file layout as the 2D solvers.
    // movie_frames spreads roughly that many snapshots over the run
    // (including initial and final); movie_stride overrides it when positive;
    // both zero disables snapshots and the run saves only the final state.
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
        // A run owns its snapshot series: clear leftovers so frames from an
        // earlier run can never leak into this run's movie.
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
        std::cerr << "ICRF_1D: time advancement stopped before the requested "
                     "final time.\n";
        return 2;
    }

    return EXIT_SUCCESS;
}

