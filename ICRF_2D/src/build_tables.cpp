/**
 * build_tables.cpp
 *
 * Stores dimensionless quasilinear resonance geometry:
 *
 *   res(x0, theta0)
 *
 * The solver reconstructs B_ql, C_ql, E_ql, F_ql from res after
 * interpolation, preserving E = sin(theta0) C and B F = C E at solver nodes.
 *
 * Compile (Linux):  g++ -O3 -std=c++17 -fopenmp build_tables.cpp -o build_tables
 * Compile (macOS):  g++ -O3 -std=c++17            build_tables.cpp -o build_tables
 */
#include <iostream>
#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "asgard.hpp"
#include "GL.hpp"


#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;
static constexpr double EPS0 = 8.8541878128e-12;
static constexpr double AMU_KG = 1.66053906660e-27;
static constexpr double KEV_J = 1.602176634e-16;
static constexpr double DENSITY_UNIT = 1.0e20;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Numerical helpers
// ═══════════════════════════════════════════════════════════════════════════

static double brentq(std::function<double(double)> f,
                     double a, double b, double tol=1e-12)
{
    double fa=f(a),fb=f(b);
    if(fa*fb>0.0) throw std::runtime_error("brentq: same sign");
    double c=a,fc=fa,d=0; bool mflag=true; double s=b,fs;
    for(int it=0;it<200;++it){
        if(std::abs(b-a)<tol) return s;
        if(fa!=fc&&fb!=fc)
            s=(a*fb*fc/((fa-fb)*(fa-fc))+b*fa*fc/((fb-fa)*(fb-fc))+c*fa*fb/((fc-fa)*(fc-fb)));
        else s=b-fb*(b-a)/(fb-fa);
        double lo=std::min((3*a+b)/4.0,b),hi=std::max((3*a+b)/4.0,b);
        bool cond=!(lo<s&&s<hi)||(mflag&&std::abs(s-b)>=std::abs(b-c)/2)||
                  (!mflag&&std::abs(s-b)>=std::abs(c-d)/2)||
                  (mflag&&std::abs(b-c)<tol)||(!mflag&&std::abs(c-d)<tol);
        if(cond){s=(a+b)/2.0;mflag=true;}else mflag=false;
        d=c;c=b;fc=fb;fs=f(s);
        if(fa*fs<0){b=s;fb=fs;}else{a=s;fa=fs;}
        if(std::abs(fa)<std::abs(fb)){std::swap(a,b);std::swap(fa,fb);}
    }
    return s;
}

// PCHIP slope at the middle sample of two adjacent, equally spaced cells.
// A sign change marks a resolved extremum, where a zero slope prevents the
// interpolant from inventing an additional extremum.  On a monotone segment,
// the harmonic mean limits a steep neighboring secant from dominating.
static inline double pchip_slope(double delta_left,double delta_right)
{
    if(delta_left*delta_right<=0.0) return 0.0;
    return 2.0*delta_left*delta_right/(delta_left+delta_right);
}

// A centered slope is safe for both cells adjacent to its node when it has
// their common sign and is no larger than three times either secant.  Then
// each normalized endpoint slope lies in [0,3], a sufficient condition for a
// monotone cubic on monotone data.  Otherwise use the PCHIP-limited slope.
static inline bool centered_slope_is_safe(double delta_left,
                                          double delta_right,
                                          double centered)
{
    if(delta_left*delta_right<=0.0) return centered==0.0;
    return centered*delta_left>0.0 &&
           std::abs(centered)<=3.0*std::min(std::abs(delta_left),
                                            std::abs(delta_right));
}

// Final hybrid slope at one periodic grid node.  A Surface computes this once
// while building its cached cubic table, rather than repeating the safety test
// at every interpolation query.
static inline double hybrid_slope(double delta_left,double delta_right)
{
    const double centered=0.5*(delta_left+delta_right);
    return centered_slope_is_safe(delta_left,delta_right,centered)
        ? centered : pchip_slope(delta_left,delta_right);
}

// Precomputed endpoint values and final hybrid slopes for every periodic grid
// cell.  cell[j] stores {f_j, f_{j+1}, m_j, m_{j+1}} for local t in [0,1].
// Runtime interpolation therefore performs only the cubic evaluation.
struct CubicTable {
    std::vector<std::array<double,4>> cell;
};

// Build one periodic interpolation table.  Each node keeps its centered slope
// when that slope is monotonicity-safe; otherwise it receives the PCHIP slope.
static CubicTable build_cubic_table(const std::vector<double>& values)
{
    const int N=static_cast<int>(values.size());
    std::vector<double> slope(N);
    for(int j=0;j<N;++j){
        const int jm=(j-1+N)%N,jp=(j+1)%N;
        slope[j]=hybrid_slope(values[j]-values[jm],
                              values[jp]-values[j]);
    }

    CubicTable table;
    table.cell.resize(N);
    for(int j=0;j<N;++j){
        const int jp=(j+1)%N;
        const double m0=slope[j],m1=slope[jp];
        table.cell[j]={values[j],values[jp],m0,m1};
    }
    return table;
}

// Cubic Hermite polynomial on one cell.  The slopes use grid-index units:
// on this uniform grid, m = df/dj = (df/dtheta) Delta-theta.
static inline double cubic_hermite(double f0,double f1,
                                   double m0,double m1,double t)
{
    double t2=t*t, t3=t2*t;
    return (2*t3-3*t2+1)*f0 + (t3-2*t2+t)*m0
         + (-2*t3+3*t2)*f1  + (t3-t2)*m1;
}

// Derivative of cubic_hermite with respect to its local coordinate t.  Surface
// converts this to d/dtheta using the uniform angular cell width.
static inline double cubic_hermite_dt(double f0,double f1,
                                      double m0,double m1,double t)
{
    const double t2=t*t;
    return (6*t2-6*t)*f0 + (3*t2-4*t+1)*m0
         + (-6*t2+6*t)*f1 + (3*t2-2*t)*m1;
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Solov'ev equilibrium
// ═══════════════════════════════════════════════════════════════════════════

// Analytic Solov'ev equilibrium used to define psi(R,Z), its derivatives, and
// the toroidal-field function F(psi).  It owns equilibrium parameters only;
// a Surface below samples one particular closed psi contour from it.
struct Equilibrium {
    // User inputs: reference major radius, inverse aspect ratio, elongation,
    // triangularity, reference field, and the linear F^2(psi) coefficient.
    double R0,eps,kappa,delta,B0,alpha;

    // Derived boundary locations and analytic coefficients.  Here A and B are
    // coefficients in psi(R,Z); B is not the magnetic-field magnitude.
    double a,R1,R2,Rt,Zt,C,A,B,F0sq;

    Equilibrium(double R0_,double eps_,double kappa_,double delta_,
                double B0_,double alpha_)
        :R0(R0_),eps(eps_),kappa(kappa_),delta(delta_),B0(B0_),alpha(alpha_)
    {
        a=eps*R0; R1=R0*(1-eps); R2=R0*(1+eps);
        Rt=R0*(1-delta*eps); Zt=kappa*a;
        C=8.0/(R0*R0*R0*R0);
        A=-(C/8*R1*R1*R1*R1)/(R1*R1*std::log(R1)-R1*R1/2);
        B=-(C/8*Rt*Rt*Rt*Rt+A*(Rt*Rt*std::log(Rt)-Rt*Rt/2))/(Zt*Zt);
        F0sq=R0*B0*R0*B0;
    }
    // Poloidal flux and the derivatives needed for B_pol = |grad psi|/R.
    double psi(double R,double Z) const {
        return C/8*R*R*R*R+A*(R*R*std::log(R)-R*R/2)+B*Z*Z; }
    double psiR(double R,double) const { return C/2*R*R*R+2*A*R*std::log(R); }
    double psiZ(double,double Z) const { return 2*B*Z; }
    // Radial curvature of psi, used with psi_ZZ=2B to classify stationary
    // points as an O-point or an X-point.
    double psiRR(double R) const {
        return 1.5*C*R*R+2*A*(std::log(R)+1.0); }
    // Toroidal-field function F=R B_phi on flux surface p.
    double F_of_psi(double p) const {
        double v=F0sq+2*alpha*p; return std::sqrt(v>1e-10?v:1e-10); }

    // The magnetic-axis O-point and midplane X-point that bound the normalized
    // flux coordinate used by Surface.
    struct CriticalPoints {
        double R_axis;
        double R_x;
    };
    // Find both positive-R stationary points analytically bracketed on either
    // side of Rcrit, then classify them from the Hessian determinant.
    CriticalPoints critical_points() const {
        // For R>0, radial stationary points solve
        //     C R^2 + 4 A log(R) = 0.
        // With C>0 and A<0 this function has one minimum, so a valid
        // O/X topology has exactly one root on each side of R_crit.
        if(!(C>0.0) || !(A<0.0) || B==0.0)
            throw std::runtime_error(
                "The equilibrium does not have a nondegenerate two-root O/X topology.");

        auto stationary_eq=[&](double R){return C*R*R+4.0*A*std::log(R);};
        const double Rcrit=std::sqrt(-2.0*A/C);
        if(!(stationary_eq(Rcrit)<0.0))
            throw std::runtime_error(
                "The radial stationary points are absent or degenerate.");

        double Rlo=1e-12*std::max(1.0,R0);
        for(int i=0;i<12 && stationary_eq(Rlo)<=0.0;++i) Rlo*=0.1;
        if(!(stationary_eq(Rlo)>0.0))
            throw std::runtime_error("Could not bracket the inner stationary point.");

        double Rhi=std::max({2.0*Rcrit,2.0*R0,R2,Rt});
        for(int i=0;i<32 && stationary_eq(Rhi)<=0.0;++i) Rhi*=2.0;
        if(!(stationary_eq(Rhi)>0.0))
            throw std::runtime_error("Could not bracket the outer stationary point.");

        const double Rin=brentq(stationary_eq,Rlo,Rcrit);
        const double Rout=brentq(stationary_eq,Rcrit,Rhi);
        const double det_in=psiRR(Rin)*(2.0*B);
        const double det_out=psiRR(Rout)*(2.0*B);

        if(det_in>0.0 && det_out<0.0) return {Rin,Rout};
        if(det_out>0.0 && det_in<0.0) return {Rout,Rin};
        throw std::runtime_error(
            "Could not classify exactly one O-point and one X-point.");
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 3. Flux surface
// ═══════════════════════════════════════════════════════════════════════════


// One closed flux surface sampled uniformly in polar angle about the O-point.
// Besides the contour itself, this object stores magnetic/field-line geometry
// and the cached interpolants used by orbit and resonance calculations.
struct Surface {
    // Runtime surface data. Only these four angular arrays are needed after
    // construction: b=B/B_min controls orbit access, ds/dtheta is the orbit
    // length factor, and k_parallel enters the resonance mismatch.
    int N;
    std::vector<double> theta,b_norm,dsdth,kpar;

    // Runtime interpolation is required only for these three quantities.
    // b_norm=B_mag/B_min and dsdth is physical field-line distance per dtheta.
    CubicTable b_norm_cubic,dsdth_cubic,kpar_cubic;

    // The only scalar surface properties needed after construction.
    double B_min,B_max;

    // Construct the psi_frac contour (1 at the O-point, 0 at the separatrix),
    // sample its geometry, and precompute all runtime interpolation tables.
    Surface(const Equilibrium& eq,double psi_frac, double n, double m, int N_=512):N(N_)
    {
        if(!(psi_frac>0.0 && psi_frac<1.0))
            throw std::invalid_argument(
                "psi must satisfy 0 < psi < 1, with 1 at the O-point "
                "and 0 at the X-point separatrix.");

        const double PI2=2*M_PI;
        theta.resize(N);
        for(int j=0;j<N;++j) theta[j]=j*PI2/N;

        const auto critical=eq.critical_points();
        const double R_axis=critical.R_axis;
        const double R_xpoint=critical.R_x;

        const double psi_axis=eq.psi(R_axis,0.0);
        const double psi_sep=eq.psi(R_xpoint,0.0);
        const double psi_mid=psi_sep+psi_frac*(psi_axis-psi_sep);
        const double F_mid=eq.F_of_psi(psi_mid);
        const double shape_scale=std::max({eq.a,std::abs(eq.Zt),
                                           std::abs(eq.R2-R_axis),
                                           std::abs(R_axis-eq.R1)});
        const double rho_max=4.0*shape_scale;
        // These arrays are required only while constructing the contour; they
        // are intentionally local so Surface retains no unused diagnostics.
        std::vector<double> rho_tab(N,0.0);
        std::vector<double> R_contour(N),Z_contour(N);

        // -------------------------------------------------------------------
        // Stage 1: trace the requested constant-psi contour.
        //
        // A point at polar angle theta is
        //
        //     R = R_axis + rho cos(theta),   Z = rho sin(theta),
        //
        // with psi(R,Z)=psi_mid.  Every stored point is a Brent-corrected
        // radial root; predictions are used only to construct its bracket.
        // -------------------------------------------------------------------
        // Optional sign-changing rho interval produced around the continuation
        // prediction.  found=false asks the caller to use the recovery sweep.
        struct RadialBracket {
            bool found=false;
            double lo=0.0;
            double hi=0.0;
        };

        // At theta=0, Z=0 and R=R_axis+rho.  Solve this simpler midplane
        // equation once to seed the radial-root continuation.
        const auto solve_midplane=[&](){
            const auto residual=[&](double R){
                return eq.psi(R,0.0)-psi_mid;
            };
            const double f_axis=residual(R_axis);
            double R_hi;

            if(R_xpoint>R_axis){
                R_hi=R_xpoint;
            }else{
                R_hi=R_axis+shape_scale;
                for(int expansion=0;
                    expansion<32 && f_axis*residual(R_hi)>0.0;
                    ++expansion)
                    R_hi=R_axis+2.0*(R_hi-R_axis);
            }

            if(f_axis*residual(R_hi)>0.0)
                throw std::runtime_error(
                    "Could not bracket the outboard midplane surface root.");
            return brentq(residual,R_axis,R_hi)-R_axis;
        };

        // Final recovery: sweep outward and return the first sign-changing
        // radial bracket.  This selects the boundary of the region containing
        // the O-point when the local predicted bracket spans two crossings.
        const auto sweep_radial_root=[&](const auto& residual,double rho_hi,
                                         double theta){
            double lo=0.0;
            double f_lo=residual(lo);
            constexpr int Nscan=256;
            for(int k=1;k<=Nscan;++k){
                const double hi=rho_hi*static_cast<double>(k)/Nscan;
                const double f_hi=residual(hi);
                if(f_lo*f_hi<=0.0)
                    return brentq(residual,lo,hi);
                lo=hi;
                f_lo=f_hi;
            }
            throw std::runtime_error(
                "No sign-changing radial sweep bracket at theta="+
                std::to_string(theta)+", psi_frac="+
                std::to_string(psi_frac)+", rho_hi="+
                std::to_string(rho_hi));
        };

        for(int j=0;j<N;++j){
            double th=theta[j],cs=std::cos(th),sn=std::sin(th),rho;
            auto radial_residual=[&](double r){
                return eq.psi(R_axis+r*cs,r*sn)-psi_mid;
            };

            // Keep the enlarged bracket inside R>0 on inward-going rays,
            // because the Solov'ev expression contains log(R).
            double rho_hi=rho_max;
            if(cs<0.0){
                const double R_floor=1e-10*std::max(1.0,eq.R0);
                rho_hi=std::min(rho_hi,(R_axis-R_floor)/(-cs));
            }

            if(j==0){
                // Seed the continuation with the direct Z=0 solve.
                rho=solve_midplane();
            }else{
                // Continue the already corrected roots linearly.  At j=1 only
                // one root exists, so the previous root is the prediction.
                const double delta_rho=(j>=2)
                    ? rho_tab[j-1]-rho_tab[j-2]
                    : 0.0;
                const double rho_prediction=std::clamp(
                    rho_tab[j-1]+delta_rho,0.0,rho_hi);

                // Start near the prediction and double h until the residual at
                // the two bracket ends changes sign.  If an interval grows over
                // two crossings, its end signs can agree; the first-root scan
                // below is the deliberately simple recovery for that case.
                const auto predict_bracket=[&](double h_in){
                    double h=h_in;
                    for(int expansion=0;expansion<32;++expansion){
                        const double lo=std::max(0.0,rho_prediction-h);
                        const double hi=std::min(rho_hi,rho_prediction+h);
                        if(radial_residual(lo)*radial_residual(hi)<=0.0)
                            return RadialBracket{true,lo,hi};
                        if(lo==0.0 && hi==rho_hi) break;
                        h*=2.0;
                    }
                    return RadialBracket{};
                };

                const double h_in=std::max(4.0*std::abs(delta_rho),
                                            2.0*rho_hi/N);
                const RadialBracket bracket=predict_bracket(h_in);

                try{
                    rho=bracket.found
                        ? brentq(radial_residual,bracket.lo,bracket.hi)
                        : sweep_radial_root(radial_residual,rho_hi,th);
                }catch(const std::exception& e){
                    throw std::runtime_error("Flux-surface root failure at theta="+
                                             std::to_string(th)+
                                             ", psi_frac="+std::to_string(psi_frac)+
                                             ", rho_hi="+std::to_string(rho_hi)+
                                             ": "+e.what());
                }
            }
            rho_tab[j]=rho;
            R_contour[j]=R_axis+rho*cs;
            Z_contour[j]=rho*sn;
        }

        // -------------------------------------------------------------------
        // Stage 2: differentiate the closed contour geometrically.
        // -------------------------------------------------------------------
        b_norm.resize(N); kpar.resize(N); dsdth.resize(N);
        std::vector<double> dlp(N);
        std::vector<double> Bmag_on_contour(N);
        double dth=PI2/N;
        for(int j=0;j<N;++j){
            const int jp=(j+1)%N,jm=(j-1+N)%N;
            const double dR=(R_contour[jp]-R_contour[jm])/(2*dth);
            const double dZ=(Z_contour[jp]-Z_contour[jm])/(2*dth);
            dlp[j]=std::hypot(dR,dZ);
        }

        // -------------------------------------------------------------------
        // Stage 3: sample magnetic and field-line geometry on that contour.
        // -------------------------------------------------------------------
        for(int j=0;j<N;++j){
            const double pR=eq.psiR(R_contour[j],Z_contour[j]);
            const double pZ=eq.psiZ(R_contour[j],Z_contour[j]);
            const double Bphi=F_mid/R_contour[j];
            const double Bpol=std::hypot(pR,pZ)/R_contour[j];
            const double Bmag=std::hypot(Bpol,Bphi);
            Bmag_on_contour[j]=Bmag;
            const double BdgPhi=Bphi/R_contour[j];
            const double BdgTh=Bpol/dlp[j];
            dsdth[j]=Bmag/Bpol*dlp[j];
            kpar[j]=(n*BdgPhi+m*BdgTh)/Bmag;
        }
        B_min=*std::min_element(Bmag_on_contour.begin(),Bmag_on_contour.end());
        B_max=*std::max_element(Bmag_on_contour.begin(),Bmag_on_contour.end());
        for(int j=0;j<N;++j)
            b_norm[j]=Bmag_on_contour[j]/B_min;

        // Select the centered-Hermite/PCHIP slopes once, then cache each
        // cell's cubic coefficients for the runtime interpolation hot path.
        b_norm_cubic=build_cubic_table(b_norm);
        dsdth_cubic =build_cubic_table(dsdth);
        kpar_cubic  =build_cubic_table(kpar);
    }

    // Periodic interpolation location: cell i and its local coordinate t.
    struct CellCoordinate { int i; double t; };

    // Minimal grouped samples for each hot path.  Grouping shares one periodic
    // cell lookup among all quantities evaluated at the same theta.
    struct OrbitSample {
        double b;          // normalized field B/B_min
        double dsdtheta;   // field-line distance Jacobian ds/dtheta
    };
    struct ResonanceSample {
        double b;          // normalized field B/B_min
        double kparallel;  // wave number parallel to the magnetic field
    };

    // Wrap theta periodically and convert it to a cached-cell index and t.
    CellCoordinate locate(double th) const {
        const double PI2=2*M_PI;
        th=std::fmod(th,PI2); if(th<0)th+=PI2;
        const double fi=th/(PI2/N);
        const int raw=static_cast<int>(fi);
        return {raw%N,fi-raw};
    }
    // Evaluate one cached cubic at an already located coordinate.
    static double eval(const CubicTable& table,const CellCoordinate& q) {
        const auto& c=table.cell[q.i];
        return cubic_hermite(c[0],c[1],c[2],c[3],q.t);
    }
    // Differentiate one cached cubic with respect to physical theta, not its
    // unit-cell coordinate t.  This is used only for the analytic limit of an
    // exact trapped-turn resonance in the eta=0 model.
    double eval_dtheta(const CubicTable& table,const CellCoordinate& q) const {
        const auto& c=table.cell[q.i];
        return (N/(2*M_PI))*cubic_hermite_dt(c[0],c[1],c[2],c[3],q.t);
    }

    // Values shared by both L and I integrands.
    OrbitSample orbit_at(double th) const {
        const CellCoordinate q=locate(th);
        return {eval(b_norm_cubic,q),eval(dsdth_cubic,q)};
    }
    // Values needed by the resonance mismatch g(theta).
    ResonanceSample resonance_at(double th) const {
        const CellCoordinate q=locate(th);
        return {eval(b_norm_cubic,q),eval(kpar_cubic,q)};
    }
    // Individual accessors used when only one interpolated quantity is needed.
    double dsdth_at(double th) const {
        const CellCoordinate q=locate(th);return eval(dsdth_cubic,q);
    }
    double b_at(double th) const {
        const CellCoordinate q=locate(th);return eval(b_norm_cubic,q);
    }
    double b_dtheta_at(double th) const {
        const CellCoordinate q=locate(th);return eval_dtheta(b_norm_cubic,q);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// 4. Turn/orbit endpoint classification
// ═══════════════════════════════════════════════════════════════════════════

// Orbit accessibility on the symmetric half contour theta in [0,pi].  It is
// computed once per pitch and reused by both L/I and resonance calculations.
struct TurnInfo {
    bool trapped=false;                 // true when a bounce point exists
    double theta_end=M_PI;              // pi if passing; first bounce angle if trapped
    int last_full_cell=0;                // last complete accessible theta cell
    bool has_partial_turn_cell=false;   // whether a final truncated cell exists
    int turn_cell=-1;                   // cell containing theta_end, or -1 if passing
};

// Classify the orbit from sin^2(pitch) and locate the first trapped-particle
// solution of b(theta) sin^2(pitch)=1.  A negative theta_end marks a degenerate
// tangent/extreme case for which no isolated bracketed endpoint exists.
static TurnInfo compute_turn(const Surface& S,double sin2pa)
{
    TurnInfo turn;
    const int Nhalf=S.N/2;
    turn.trapped=sin2pa*(S.B_max/S.B_min)>1.0;

    if(!turn.trapped){
        turn.theta_end=M_PI;
        turn.last_full_cell=Nhalf-1;
        return turn;
    }

    const double b_turn=1.0/sin2pa;
    for(int j=0;j<Nhalf;++j){
        if((S.b_norm[j]-b_turn)*(S.b_norm[j+1]-b_turn)<0.0){
            turn.turn_cell=j;
            turn.theta_end=brentq([&](double th){return S.b_at(th)-b_turn;},
                                  S.theta[j],S.theta[j+1]);
            turn.last_full_cell=j-1;
            turn.has_partial_turn_cell=true;
            return turn;
        }
    }

    // Tangent/extreme cases do not provide an isolated, bracketed endpoint.
    turn.theta_end=-1.0;
    turn.last_full_cell=-1;
    return turn;
}

// ---------------------------------------------------------------------------
// 5. L/I integrals
// ---------------------------------------------------------------------------
// Unnormalized geometric orbit integrals.  main applies geom_norm before
// writing them to L_tab and I_tab.
struct LI {
    double L;
    double I;
};

static LI compute_LI(const Surface& S,double pitch_angle,const TurnInfo& turn)
{
    const double sinpa=std::sin(pitch_angle);
    const double sin2pa=sinpa*sinpa;

    // Compare GL32 and GL64 first.  For a smooth interval GL64 is accepted;
    // only a measured disagreement pays for the GL128 fallback.
    auto integrate_checked=[&](auto const& f,double a,double b)->double{
        double g32 = gl::integrate<32>(f,a,b);
        double g64 = gl::integrate<64>(f,a,b);
        double scale = std::max(std::abs(g32),std::abs(g64));
        if(scale<1e-300 || std::abs(g32-g64) <= 1e-6*scale)
            return g64;
        return gl::integrate<128>(f,a,b);
    };

    // Work on the symmetric half-orbit [0, theta_last].  For passing particles
    // the last physical point is pi, where B is maximal but still below the
    // turning value.  For trapped particles it is the true turning point.  The
    // same map theta = theta_last - u^2 clusters quadrature nodes at this sharp
    // endpoint and removes the trapped inverse-square-root singularity.
    const double theta_last=turn.theta_end;
    if(theta_last<=0.0) return {0.0,0.0};
    // A trapped half-orbit represents four symmetric pieces; a passing
    // half-contour represents two.
    const double orbit_symmetry=turn.trapped ? 4.0 : 2.0;

    const double u_end=std::sqrt(theta_last);
    auto L_integrand_u=[&](double u)->double{
        double th=theta_last-u*u;
        const auto sample=S.orbit_at(th);
        double arg=1.0-sin2pa*sample.b;
        return arg<=1e-15?0.0:2.0*u*sample.dsdtheta/std::sqrt(arg);};
    
    auto I_integrand_u=[&](double u)->double{
        double th=theta_last-u*u;
        const auto sample=S.orbit_at(th);
        double arg=1.0-sin2pa*sample.b;
        return arg<=0?0.0:
            2.0*u*sample.dsdtheta*std::sqrt(arg)/sample.b;};

    return {orbit_symmetry*integrate_checked(L_integrand_u,0.0,u_end),
            orbit_symmetry*integrate_checked(I_integrand_u,0.0,u_end)};
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Resonance scan interval and resonance summation
//
//   res = sum_R  b_R / (|cosθ_R| |g'(s_R)|)
//   The solver multiplies this finite resonance sum by sin²(theta0).
//   where g = k_parallel vta x_parallel - (omega - Omega_s0 b).
//   The solver reconstructs B_ql, C_ql, E_ql, F_ql from res and L.
// ═══════════════════════════════════════════════════════════════════════════

// Everything below this point is the calculation for one directed orbit
// branch, sigma=sign(v_parallel).  Group the immutable physics together so
// the exact and finite-linewidth paths do not each carry a long argument list.
// `turn` is shared across both trapped branches because it describes geometry,
// not the direction of parallel motion.
struct ResonanceBranch {
    const Surface& surface;
    const TurnInfo& turn;
    double speed;       // v = x * v_ta
    double sin2_pitch;  // sin^2(theta0)
    double omega;
    double Omega_s0;
    double gamma;
    double sigma;       // +1 or -1 branch of v_parallel

    double integrate() const;
};

double ResonanceBranch::integrate() const
{
    // Short aliases make the formulas below read like their mathematical
    // notation while the owning ResonanceBranch keeps all shared state visible
    // in one declaration above.
    const Surface& S=surface;
    const double v_speed=speed;
    const double sin2pa=sin2_pitch;
    const int N=S.N;
    const double dth = 2*M_PI/N;

    // Pitch = exactly 90 deg has no passing region at the midplane minimum:
    // vp^2 = v^2(1 - b_norm sin^2(theta0)) is zero at the minimum-B point and
    // negative elsewhere.  Since v_parallel is not real, there is no branch to
    // scan in this delta-function resonance approximation.
    // No passing region exists iff vp2 = v^2(1 - b_norm*sin2pa) <= 0 everywhere,
    // i.e. min_j b_norm[j]*sin2pa >= 1.  Since min_j b_norm = 1 by construction,
    // this holds only at pitch = exactly 90 deg.
    if(sin2pa >= 1.0) return 0.0;

    const bool trapped=turn.trapped;
    const double theta_end = turn.theta_end;
    const int last_full_cell = turn.last_full_cell;
    const bool has_partial_turn_cell = turn.has_partial_turn_cell;
    const int turn_cell = turn.turn_cell;
    if(trapped && (turn_cell<1 || theta_end<=0.0)) return 0.0;

    // Resonance mismatch at an arbitrary position on one direction branch:
    //
    //     g(theta) = k_parallel(theta) v_parallel(theta)
    //              - [omega - Omega_s0 b(theta)].
    //
    // sigma = +1/-1 selects the sign of v_parallel.  Points beyond a trapped
    // turning point have negative v_parallel^2, so the mismatch is undefined;
    // return NaN to make every scanner skip those points.
    auto mismatch_at = [&](double th, double sigma)->double{
        const auto sample=S.resonance_at(th);
        double b = sample.b;
        double arg = 1.0 - b*sin2pa;
        if(arg < -1e-12) return std::numeric_limits<double>::quiet_NaN();
        double vpar = sigma * v_speed * std::sqrt(std::max(0.0, arg));
        return sample.kparallel*vpar - (omega - Omega_s0*b);
    };

    // Compute dg/ds at the resonant point for the delta-function weight.
    // We differentiate in theta with a local finite difference and convert
    // using ds/dtheta.  Near a branch endpoint, switch to a one-sided stencil so
    // the derivative never samples outside the physical interval [0, upper].
    auto mismatch_s_derivative_at = [&](double th, double sigma,
                                        double upper)->double{
        double h = 0.25*dth;
        double lo = 0.0, hi = upper;
        double gm, gp, dgdth;
        if(th-h >= lo && th+h <= hi){
            gm = mismatch_at(th-h, sigma);
            gp = mismatch_at(th+h, sigma);
            dgdth = (gp-gm)/(2.0*h);
        }else if(th+h <= hi){
            gm = mismatch_at(th, sigma);
            gp = mismatch_at(th+h, sigma);
            dgdth = (gp-gm)/h;
        }else if(th-h >= lo){
            gm = mismatch_at(th-h, sigma);
            gp = mismatch_at(th, sigma);
            dgdth = (gp-gm)/h;
        }else{
            return std::numeric_limits<double>::quiet_NaN();
        }
        double dsdth = S.dsdth_at(th);
        return dsdth>1e-30 ? dgdth/dsdth : std::numeric_limits<double>::quiet_NaN();
    };

    // Locate a zero inside a complete grid cell [theta[j], theta[j+1]].
    // A sign change has already bracketed the root.  Try inverse quadratic
    // interpolation with the previous point for a sharper root estimate, but
    // accept it only if it remains inside the bracket.  Otherwise fall back to
    // linear interpolation, which is bracket-safe.
    auto estimate_root_in_full_cell = [&](int j,
                                          const std::vector<double>& mismatch)->double{
        double th_a=S.theta[j], th_b=S.theta[j+1];
        double ga=mismatch[j], gb=mismatch[j+1];
        bool use_iqi=false;
        double th_R=0.0;
        if(j>=1 && std::isfinite(mismatch[j-1])){
            double tc=S.theta[j-1], gc=mismatch[j-1];
            if(ga!=gb && ga!=gc && gb!=gc){
                double th_q = th_a*gb*gc/((ga-gb)*(ga-gc))
                            + th_b*ga*gc/((gb-ga)*(gb-gc))
                            + tc  *ga*gb/((gc-ga)*(gc-gb));
                if(th_q>th_a && th_q<th_b){ th_R=th_q; use_iqi=true; }
            }
        }
        return use_iqi ? th_R : th_a - ga*(th_b-th_a)/(gb-ga);
    };

    // Add one resonant point to the geometric sum.  At resonance,
    //
    //     cos(theta_R)^2 = 1 - b(theta_R) sin^2(theta0),
    //
    // and the delta-function reduction contributes b_R/(|cos(theta_R)| |dg/ds|).
    // Degenerate/tangent cases have nearly zero denominator and are skipped here;
    // those require a different tangent-resonance treatment, not this simple
    // isolated-root formula.
    auto exact_root_weight = [&](double th_R, double sigma,
                                 double upper)->double{
        double dgds_R = mismatch_s_derivative_at(th_R, sigma, upper);
        double b_R    = S.b_at(th_R);
        double sin2_R = b_R * sin2pa;
        double cos_R  = sin2_R<1.0 ? std::sqrt(1.0-sin2_R) : 0.0;
        double denom  = std::abs(cos_R * dgds_R);
        if(!(denom>1e-30)) return 0.0;   // also skips NaN
        return b_R / denom;
    };

    // One sign-changing mismatch interval found by the coarse orbit scan.
    //
    // The scan itself is intentionally cheap: it uses the stored surface grid
    // and only establishes that one isolated resonance lies in [theta_lo,
    // theta_hi].  The eta=0 model uses this bracket directly for its root
    // estimate.  The finite-linewidth model refines it with Brent only once,
    // then places a dedicated high-order quadrature panel around that root.
    // `grid_cell` and `full_grid_cell` distinguish an ordinary stored grid cell from the
    // shortened final cell of a trapped orbit, whose physical upper endpoint
    // is theta_end rather than theta[cell+1].
    struct RootBracket {
        int grid_cell;
        double theta_lo,theta_hi;
        double mismatch_lo,mismatch_hi;
        bool full_grid_cell;
    };

    // Scan one sign branch of v_parallel once. Both resonance models share
    // these sampled mismatch values and sign-change brackets. Only their final
    // accumulation differs: an exact delta-function Jacobian sum for gamma=0,
    // or a finite-width orbit integral for gamma>0.
    auto scan_branch = [&](double sigma)->double{
        // `mismatch[j]` is g(theta[j]) on this parallel-velocity branch.
        // NaN marks a point beyond a trapped turning point.
        std::vector<double> mismatch(N);

        // Precompute g on the stored grid.  Unreachable points are marked NaN;
        // this lets the complete-cell loop ignore cells beyond a turning point.
        for(int j=0;j<N;++j){
            // Orbit accessibility is geometric and remains meaningful at
            // v_speed=0.  Testing v_parallel^2 instead would mark every point
            // unreachable at zero speed and artificially force the first
            // resonance-table row to zero.
            const double arg = 1.0-S.b_norm[j]*sin2pa;
            if(arg>=0.0){
                const double vpar = sigma*v_speed*std::sqrt(arg);
                mismatch[j]=S.kpar[j]*vpar - (omega - Omega_s0*S.b_norm[j]);
            }else{
                mismatch[j]=std::numeric_limits<double>::quiet_NaN();
            }
        }

        std::vector<RootBracket> root_brackets;

        // Complete cells: both endpoints lie inside the physical branch.
        for(int j=0;j<=last_full_cell;++j){
            if(!std::isfinite(mismatch[j])||!std::isfinite(mismatch[j+1])) continue;
            if(mismatch[j]*mismatch[j+1]<0)
                root_brackets.push_back({j,S.theta[j],S.theta[j+1],
                                         mismatch[j],mismatch[j+1],true});
        }

        // Final partial cell for trapped orbits.  The actual branch endpoint is
        // theta_end, not theta[turn_cell+1]; at theta[turn_cell+1] v_parallel is
        // undefined.  Check only the physical interval theta[turn_cell] -> theta_end.
        if(has_partial_turn_cell && std::isfinite(mismatch[turn_cell])){
            const double mismatch_end=mismatch_at(theta_end, sigma);
            if(std::isfinite(mismatch_end)
               && mismatch[turn_cell]*mismatch_end<0.0)
                root_brackets.push_back({turn_cell,S.theta[turn_cell],theta_end,
                                         mismatch[turn_cell],mismatch_end,false});
        }

        // A strict sign-change scan deliberately does not count a zero that
        // lands exactly on either physical endpoint.  Check those two points
        // explicitly instead of treating either adjacent grid cell as a second
        // root bracket.  The tolerance is round-off scaled by the local
        // mismatch magnitude, so this detects only an endpoint root, not a
        // merely nearby finite-linewidth peak.
        const double endpoint_scale=1.0+std::abs(omega)
            +std::abs(Omega_s0*S.b_at(theta_end))
            +std::abs(S.kpar[0]*v_speed);
        const double endpoint_tolerance=128.0
            *std::numeric_limits<double>::epsilon()*endpoint_scale;
        const double g_midplane=mismatch_at(0.0,sigma);
        const double g_upper=mismatch_at(theta_end,sigma);
        const bool resonance_at_midplane=std::isfinite(g_midplane)
            && std::abs(g_midplane)<=endpoint_tolerance;
        const bool resonance_at_upper=std::isfinite(g_upper)
            && std::abs(g_upper)<=endpoint_tolerance;

        // ── Model A: exact delta-function reduction (gamma = 0) ──────────
        //
        // The shared scan already found every sign-changing resonance cell.
        // This model needs only an isolated-root estimate and its Jacobian;
        // it never performs an orbit quadrature.
        auto integrate_exact_delta = [&]()->double {
            double res_branch=0.0;
            for(RootBracket const& root : root_brackets){
                const double theta_root = root.full_grid_cell
                    ? estimate_root_in_full_cell(root.grid_cell,mismatch)
                    : root.theta_lo-root.mismatch_lo
                        *(root.theta_hi-root.theta_lo)
                        /(root.mismatch_hi-root.mismatch_lo);
                res_branch += exact_root_weight(theta_root,sigma,theta_end);
            }

            // theta=0 and theta=pi are symmetry points of this equilibrium.
            // An exact resonance there is tangent: nu_u=0, so the eta=0 delta
            // reduction has no finite simple-root limit.  A finite linewidth
            // is physically and numerically required instead of a fabricated
            // half-weight endpoint contribution.
            if(resonance_at_midplane || (resonance_at_upper && !trapped))
                throw std::runtime_error(
                    "eta=0 tangent resonance at a symmetry endpoint; use eta>0.");

            if(trapped && resonance_at_upper){
                // At a normal trapped turning point, theta=theta_turn-u^2,
                // so
                //
                //   1-b sin^2(theta0) = A u^2 + O(u^4),
                //   A = sin^2(theta0) db/dtheta|turn > 0.
                //
                // The transformed prefactor P(u) and nu_u(0) are both finite.
                // The one-sided delta integral is therefore
                //
                //   1/2 P(0)/|nu_u(0)|
                //   = b_turn (ds/dtheta)_turn / (|k_parallel| v A).
                //
                // This is the finite limit of the usual interior-root weight,
                // evaluated directly from the cached cubic derivative rather
                // than as the indeterminate product cos(theta_R)*dnu/ds.
                const auto turn_sample=S.resonance_at(theta_end);
                const double A=sin2pa*S.b_dtheta_at(theta_end);
                const double denominator=std::abs(turn_sample.kparallel)
                    *v_speed*A;
                if(!(A>0.0) || !(denominator>1.0e-30))
                    throw std::runtime_error(
                        "eta=0 trapped-turn resonance is tangent; use eta>0.");
                res_branch+=turn_sample.b*S.dsdth_at(theta_end)/denominator;
            }
            return res_branch;
        };
        if(gamma==0.0) return integrate_exact_delta();

        // ── Model B: finite-linewidth orbit integral (gamma > 0) ─────────
        //
        // This is the normal production path when a finite linewidth is
        // requested.  Unlike the eta=0 delta-function path above, it must
        // integrate the Lorentzian over the *whole* accessible orbit.  The
        // roots found here only tell us where to place expensive, centred
        // quadrature panels; they do not replace the orbit integral.
        //
        // Use theta = theta_end-u^2.  For trapped particles this cancels the
        // square-root factor at the turning point theta_end.  The integration
        // coordinate therefore always runs in the simple direction
        //
        //     u=0  (turning point, or theta=pi for passing particles)
        //     u=u_max=sqrt(theta_end)  (theta=0 midplane).
        //
        // First refine each ordinary sign-changing theta bracket into the
        // centre u_R of one Lorentzian peak.
        auto integrate_broadened = [&]()->double {
        std::vector<double> roots_u;
        roots_u.reserve(root_brackets.size());
        for(RootBracket const& root : root_brackets){
            // `root` contains a physically accessible, sign-changing cell
            // from the inexpensive scan. Brent is safe here because it cannot
            // leave this bracket or enter an inaccessible part of the orbit.
            const double theta_root=brentq(
                [&](double th) { return mismatch_at(th,sigma); },
                root.theta_lo,root.theta_hi);
            // Convert theta_R into the coordinate used by all panels below.
            // max(...,0) removes a possible negative round-off at theta_end.
            const double u_root=std::sqrt(std::max(0.0,theta_end-theta_root));
            // A finite u_R is an interior peak centre. Exact endpoint roots
            // are deliberately handled separately below, not inserted here.
            if(std::isfinite(u_root)) roots_u.push_back(u_root);
        }

        // The finite-linewidth branch integrates the Lorentzian replacement
        // for delta(nu), where
        //
        //   delta(nu) -> gamma / [pi (nu^2 + gamma^2)].
        //
        // Starting from the orbit factor
        //
        //   b (ds/dtheta) / sqrt(1-b sin^2(theta0)) dtheta,
        //
        // theta=theta_end-u^2 gives dtheta=-2u du.  Reversing the interval
        // yields the positive Jacobian 2u below.  This form is regular at a
        // trapped turning point: u cancels the original square-root behavior.
        auto integrand_u = [&](double u)->double {
            const double th=theta_end-u*u;
            const auto sample=S.resonance_at(th);
            const double arg=1.0-sample.b*sin2pa;
            if(arg<=0.0) return 0.0;

            // nu is the signed local resonance mismatch.  It vanishes at a
            // resonance and is otherwise sampled everywhere because a finite
            // Lorentzian has nonzero tails away from its centre.
            const double nu=sample.kparallel*sigma*v_speed*std::sqrt(arg)
                          - (omega-Omega_s0*sample.b);
            const double kernel=gamma/(M_PI*(nu*nu+gamma*gamma));
            return 2.0*u*sample.b*S.dsdth_at(th)*kernel/std::sqrt(arg);
        };

        // The far integration endpoint corresponds exactly to theta=0.
        const double u_max=std::sqrt(theta_end);

        // Reuse the physical mismatch in the transformed coordinate.  This
        // is used only to determine the local Lorentzian width around u_R.
        auto mismatch_u = [&](double u)->double {
            return mismatch_at(theta_end-u*u,sigma);
        };

        // Estimate dnu/du at a root.  Its reciprocal converts the physical
        // linewidth gamma into a local u-width gamma/|dnu/du|.  Use a centred
        // difference in the ordinary interior case.  At u=0 or u=u_max use a
        // one-sided difference so no sample ever leaves the physical orbit.
        // The step scales with u_max but has an absolute floor, avoiding a
        // subtraction of almost equal floating-point values on short orbits.
        auto dnud_u = [&](double u)->double {
            const double h0=std::max(1.0e-8,1.0e-5*std::max(1.0,u_max));
            const double h=std::min(h0,0.25*std::max(u_max,1.0e-12));
            if(u-h>=0.0 && u+h<=u_max)
                return (mismatch_u(u+h)-mismatch_u(u-h))/(2.0*h);
            if(u+h<=u_max) return (mismatch_u(u+h)-mismatch_u(u))/h;
            if(u-h>=0.0) return (mismatch_u(u)-mismatch_u(u-h))/h;
            return std::numeric_limits<double>::quiet_NaN();
        };

        // Sort only the ordinary interior roots in integration order.  An
        // endpoint cannot be the centre of a symmetric panel, so endpoint
        // roots are deliberately kept out of this list and treated below.
        std::sort(roots_u.begin(),roots_u.end());
        // A simple Lorentzian has u half-width gamma/|dnu/du|.  The high-order
        // panel covers sixteen such widths.  This number is named rather than
        // hidden so the resolved fraction of each Lorentzian is explicit.
        constexpr double linewidths_in_peak_panel=16.0;
        const double roundoff_panel_width=32.0
            *std::numeric_limits<double>::epsilon()*std::max(1.0,u_max);

        // Choose the width of a one-sided endpoint panel.  `width_available` is the
        // part of the orbit reserved for that endpoint before the nearest
        // interior-root cell begins.  Thus this panel can never overlap a
        // centred panel.  For a simple root, its requested width is sixteen
        // Lorentzian half-widths.  If dnu/du vanishes, the endpoint is tangent
        // and no simple-root linewidth exists, so integrate all available
        // space at high order instead of dividing by a near-zero slope.
        auto endpoint_width = [&](double endpoint,double width_available)->double {
            if(!(width_available>0.0)) return 0.0;
            const double slope=std::abs(dnud_u(endpoint));
            if(!(slope>0.0) || !std::isfinite(slope)) return width_available;
            const double physical=linewidths_in_peak_panel*gamma/slope;
            return std::min(width_available,
                            std::max(physical,roundoff_panel_width));
        };

        // ── Endpoint-root edge cases ──────────────────────────────────────
        //
        // Exact roots at theta_end (u=0) or theta=0 (u=u_max) are rare. They
        // cannot own a symmetric panel, so give them a one-sided GL128 panel.
        // Reserve at most half of the distance to the nearest interior root;
        // if both endpoints resonate and there is no interior root, each gets
        // at most half of the entire orbit.  This prevents double counting.
        const double upper_theta_endpoint_available=roots_u.empty()
            ? (resonance_at_midplane ? 0.5*u_max : u_max)
            : 0.5*roots_u.front();
        const double midplane_theta_endpoint_available=roots_u.empty()
            ? (resonance_at_upper ? 0.5*u_max : u_max)
            : 0.5*(u_max-roots_u.back());

        // Accumulate endpoint panels first.  These two coordinates then mark
        // the unintegrated middle interval.  Every regular root cell below
        // lies entirely inside [u_after_upper_theta_endpoint,
        //                    u_before_midplane_theta_endpoint].
        double res_branch=0.0;
        double u_after_upper_theta_endpoint=0.0;
        if(resonance_at_upper){
            const double width=endpoint_width(0.0,upper_theta_endpoint_available);
            res_branch+=gl::integrate<128>(integrand_u,0.0,width);
            u_after_upper_theta_endpoint=width;
        }

        double u_before_midplane_theta_endpoint=u_max;
        if(resonance_at_midplane){
            const double width=endpoint_width(u_max,midplane_theta_endpoint_available);
            u_before_midplane_theta_endpoint=u_max-width;
            res_branch+=gl::integrate<128>(integrand_u,u_before_midplane_theta_endpoint,
                                            u_max);
        }

        if(roots_u.empty()){
            // Main no-interior-root case: after any endpoint panel, the
            // remaining interval contains only smooth Lorentzian tails.  No
            // permanent turning-point treatment is needed because the u map
            // already removed the trapped-turn square-root behaviour.  Use
            // GL128 deliberately: finite eta table generation prioritizes a
            // converged resonance integral over the modest extra cost.
            if(u_before_midplane_theta_endpoint>u_after_upper_theta_endpoint)
                res_branch+=gl::integrate<128>(integrand_u,u_after_upper_theta_endpoint,
                                               u_before_midplane_theta_endpoint);
            return res_branch;
        }

        // ── Ordinary interior-root case ────────────────────────────────────
        //
        // Give every root one coarse cell.  The cell boundaries are midpoints
        // between adjacent roots, except at the physical orbit ends where
        // they meet the already-computed endpoint-panel boundary.  Thus all
        // cells exactly tile the remaining orbit: no overlap, gaps, cursor,
        // window merging, or implicit bookkeeping is required.
        auto integrate_root_cell = [&](double root,double cell_left,
                                       double cell_right)->double {
            // Distances from the root to its own coarse-cell boundaries.
            const double left_distance=root-cell_left;
            const double right_distance=cell_right-root;
            // Degenerate cells should not occur for distinct sign-changing
            // roots. If round-off makes one occur, integrate that whole cell
            // safely at high order rather than constructing a negative width.
            if(!(left_distance>0.0) || !(right_distance>0.0))
                return gl::integrate<128>(integrand_u,cell_left,cell_right);

            // A tangent root has no useful simple-root linewidth. Treat its
            // entire cell as one high-order integral; this is an edge case.
            const double slope=std::abs(dnud_u(root));
            if(!(slope>0.0) || !std::isfinite(slope))
                return gl::integrate<128>(integrand_u,cell_left,cell_right);

            // The physical Lorentzian half-width in u.  The central panel
            // covers sixteen widths, then is clipped to its own cell so it
            // remains centred at the resonance and never overlaps a neighbour.
            const double physical_half_width=
                linewidths_in_peak_panel*gamma/slope;
            const double peak_half_width=std::min(
                {left_distance,right_distance,
                 std::max(physical_half_width,roundoff_panel_width)});
            const double peak_left=root-peak_half_width;
            const double peak_right=root+peak_half_width;

            // The central interval contains the sharp Lorentzian peak.  The
            // left/right intervals are smooth tails.  All three use GL128 in
            // the finite-linewidth model so the full resonance integral is
            // uniformly high-order, not only the central peak.
            double value=0.0;
            if(peak_left>cell_left)
                value+=gl::integrate<128>(integrand_u,cell_left,peak_left);
            value+=gl::integrate<128>(integrand_u,peak_left,peak_right);
            if(cell_right>peak_right)
                value+=gl::integrate<128>(integrand_u,peak_right,cell_right);
            return value;
        };

        for(std::size_t r=0;r<roots_u.size();++r){
            // Root r owns the interval halfway to the preceding and following
            // roots.  At r=0 and r=last, substitute the already-reserved
            // endpoint-panel boundary for the missing neighbour.
            const double cell_left=(r==0)
                ? u_after_upper_theta_endpoint : 0.5*(roots_u[r-1]+roots_u[r]);
            const double cell_right=(r+1==roots_u.size())
                ? u_before_midplane_theta_endpoint : 0.5*(roots_u[r]+roots_u[r+1]);
            res_branch+=integrate_root_cell(roots_u[r],cell_left,cell_right);
        }
        return res_branch;
        };
        return integrate_broadened();
    };

    // This object represents one fixed sign of v_parallel, so its only job is
    // to integrate that branch over the scanned half-contour.  Passing/trapped
    // branch selection and the final up-down symmetry are handled by the short
    // compute_res() dispatcher below.
    return scan_branch(sigma);
}

// Assemble the physical orbit from one or two directed ResonanceBranch
// objects.  The returned quantity is still the geometric resonance factor;
// the solver applies its outer sin^2(theta0) factor later.
static double compute_res(const Surface& S, double x_speed, double pitch_angle,
                          const TurnInfo& turn,
                          double omega, double Omega_s0, double vta,
                          double gamma)
{
    if(!(omega>0.0) || !(Omega_s0>0.0) || !(vta>0.0) || gamma<0.0)
        return 0.0;

    const double sin_pitch=std::sin(pitch_angle);
    const double sin2_pitch=sin_pitch*sin_pitch;

    // Exactly perpendicular launch has no real accessible parallel branch in
    // this representation. The endpoint classifier also marks degenerate
    // trapped extrema with theta_end<=0; neither case has a usable orbit.
    if(sin2_pitch>=1.0 || (turn.trapped
                            && (turn.turn_cell<1 || turn.theta_end<=0.0)))
        return 0.0;

    const double speed=vta*x_speed;
    auto branch=[&](double sigma)->double {
        return ResonanceBranch{S,turn,speed,sin2_pitch,omega,Omega_s0,
                               gamma,sigma}.integrate();
    };

    // Passing particles keep the sign selected at launch. Trapped particles
    // visit both signs during one bounce, so both directed branches contribute.
    const double half_contour_sum=turn.trapped
        ? branch(+1.0)+branch(-1.0)
        : branch(std::cos(pitch_angle)<0.0 ? -1.0 : +1.0);

    // The unscanned second half of this up-down-symmetric equilibrium gives an
    // identical contribution.
    return 2.0*half_contour_sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Save binary
// ═══════════════════════════════════════════════════════════════════════════

static void save_bin(const std::string& path, const std::vector<double>& v)
{
    FILE* f=std::fopen(path.c_str(),"wb");
    if(!f) throw std::runtime_error("Cannot write: "+path);
    std::fwrite(v.data(),sizeof(double),v.size(),f);
    std::fclose(f);
    std::printf("  saved %s  (%zu values)\n",path.c_str(),v.size());
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. Main
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    asgard::libasgard_runtime running_(argc, argv);
    asgard::prog_opts options(argc, argv);

    if (options.show_help) {
        std::cout <<
          "\n build_tables: generate L, I, and res tables\n"
          " Run with:\n"
          "     ./build_tables -if input_build.txt\n\n"
          " Required input_build keys include R0, eps, kappa, delta, B0, alpha,\n"
          " psi (1 at O-point, 0 at separatrix), omega_ratio, optional eta, n_mode, m_mode,\n"
          " species parameters, tab_Nx,\n"
          " tab_Npitch, and tab_xmax. The grid uses x_min=0 and pitch=0..180 deg.\n\n"
          "    -- standard ASGarD options --";
        options.print_help(std::cout);
        return 0;
    }

    int    N_x    = options.file_required<int>("tab_Nx");
    int    N_pitch= options.file_required<int>("tab_Npitch");
    constexpr double x_min = 0.0;
    double x_max  = options.file_required<double>("tab_xmax");
    constexpr double pa_min = 0.0;
    constexpr double pa_max = 180.0;
    // SINGLE flux surface: psi=1 at the O-point and psi=0 at the separatrix.
    const double psi_fixed = options.file_required<double>("psi");


    const double PI=M_PI;
    double pa_min_rad=pa_min*PI/180.0, pa_max_rad=pa_max*PI/180.0;

    std::printf("Grid: Nx=%d [%.2f,%.2f]  Npitch=%d [%.1f,%.1f]deg  psi_frac=%.4f\n",
                N_x,x_min,x_max, N_pitch,pa_min,pa_max, psi_fixed);
#ifdef _OPENMP
    std::printf("OpenMP: %d threads\n",omp_get_max_threads());
#else
    std::printf("OpenMP not enabled (single-threaded)\n");
#endif

    std::vector<double> x_grid(N_x), pitch_grid(N_pitch);
    for(int i=0;i<N_x;++i)
        x_grid[i]=N_x==1?x_min:x_min+i*(x_max-x_min)/(N_x-1);
    for(int i=0;i<N_pitch;++i)
        pitch_grid[i]=N_pitch==1?pa_min_rad:pa_min_rad+i*(pa_max_rad-pa_min_rad)/(N_pitch-1);


    const double B0 = options.file_required<double>("B0");
    Equilibrium eq(options.file_required<double>("R0"),
                   options.file_required<double>("eps"),
                   options.file_required<double>("kappa"),
                   options.file_required<double>("delta"),
                   B0,
                   options.file_required<double>("alpha"));
    const double omega_ratio = options.file_required<double>("omega_ratio");
    // eta is a fractional RF linewidth: gamma = eta * omega. eta=0 preserves
    // the original exact-delta root/Jacobian resonance calculation.
    const double eta_ratio = options.file_value<double>("eta").value_or(0.0);
    if(!(eta_ratio >= 0.0) || !std::isfinite(eta_ratio))
        throw std::runtime_error("eta must be finite and non-negative.");
    const double z_a = options.file_required<double>("z_a");
    const double m_a = options.file_required<double>("m_a");
    const double T_a = options.file_required<double>("T_a");
    const double logLambda_aa = options.file_required<double>("logLambda_aa");
    const double n_ref = options.file_required<double>("n_ref");

    const double vta = std::sqrt(2.0*T_a*KEV_J/(m_a*AMU_KG));
    const double m_a_kg = m_a*AMU_KG;
    const double Gamma0 = std::pow(ELEMENTARY_CHARGE, 4) * std::pow(z_a, 4)
                        * (n_ref * DENSITY_UNIT) * logLambda_aa
                        / (4.0 * M_PI * EPS0 * EPS0 * m_a_kg * m_a_kg);

    const double n = options.file_required<double>("n_mode");
    const double m = options.file_required<double>("m_mode");

    const double geom_norm = Gamma0 / std::pow(vta, 4);
    const double freq_norm = Gamma0 / std::pow(vta, 3);
    const double res_norm = geom_norm * freq_norm;


    // L, I : (N_pitch)
    std::vector<double> L_tab(N_pitch,0), I_tab(N_pitch,0);
    std::vector<TurnInfo> turn_tab(N_pitch);
    // Quasilinear resonance sum: (N_x, N_pitch), without runtime eps_E.
    std::vector<double> res_tab(N_x*N_pitch,0);

    // The single flux surface at psi_fixed: 1 at O-point, 0 at separatrix.
    std::printf("Precomputing surface at psi_frac=%.4f ...\n",psi_fixed);
    const auto surface_start = std::chrono::high_resolution_clock::now();
    Surface S(eq,psi_fixed,n,m);
    const auto surface_end = std::chrono::high_resolution_clock::now();
    const double surface_seconds =
        std::chrono::duration<double>(surface_end - surface_start).count();
    std::cout << "Took " << surface_seconds << " s.\n";
    
    const double Omega_s0 = z_a * ELEMENTARY_CHARGE * S.B_min / (m_a * AMU_KG);
    const double omega = omega_ratio * Omega_s0;
    const double gamma = eta_ratio * omega;
    if(eta_ratio == 0.0)
        std::cout << "Resonance model: exact delta-function roots.\n";
    else
        std::cout << "Resonance model: Lorentzian broadening, gamma="
                  << eta_ratio << " * omega.\n";

    // Turn/orbit endpoints (1D in pitch)
    std::printf("Building turn/orbit endpoints (%d) : \n",N_pitch);
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for(int ip=0;ip<N_pitch;++ip){
        const double sinth=std::sin(pitch_grid[ip]);
        turn_tab[ip]=compute_turn(S,sinth*sinth);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Took " << seconds << " s.\n";

    // L and I (1D in pitch)
    std::printf("Building L, I (%d) : \n",N_pitch);
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for(int ip=0;ip<N_pitch;++ip){
        LI li=compute_LI(S,pitch_grid[ip],turn_tab[ip]);
        L_tab[ip]=geom_norm*li.L;
        I_tab[ip]=geom_norm*li.I;
    }
    end = std::chrono::high_resolution_clock::now();
    seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Took " << seconds << " s.\n";

    start = std::chrono::high_resolution_clock::now();
    // res table (2D in x=v/vta, pitch)
    std::printf("Building res table (%d x %d) : \n",N_x,N_pitch);
    // Individual pitch/orbit evaluations vary substantially in cost, so
    // dynamic 2D scheduling keeps workers balanced while filling res_tab.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int ix=0;ix<N_x;++ix){
        for(int ip=0;ip<N_pitch;++ip){
            int idx=ix*N_pitch+ip;
            const double res=compute_res(S,x_grid[ix],pitch_grid[ip],turn_tab[ip],
                                         omega,Omega_s0,vta,gamma);
            res_tab[idx] = res_norm*res;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Took " << seconds << " s.\n";

    std::system("mkdir -p tables");
    std::vector<double> parameters = {
        static_cast<double>(N_x),
        static_cast<double>(N_pitch),
        x_min,
        x_max,
        pa_min_rad,
        pa_max_rad,
        omega,
        Omega_s0,
        z_a,
        m_a,
        T_a,
        logLambda_aa
    };
    save_bin("tables/parameters.bin", parameters);
    save_bin("tables/L_tab.bin",  L_tab);
    save_bin("tables/I_tab.bin",  I_tab);
    save_bin("tables/res_tab.bin", res_tab);

    std::printf("\nL,I: (%d)  res: (%d,%d)  psi_frac=%.4f\n",
                N_pitch, N_x,N_pitch, psi_fixed);
    return 0;
}
