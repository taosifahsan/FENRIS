#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <asgard.hpp>
#include <asgard_pde.hpp>

enum class pde_mode {coalesced, split, poisson, shock1d, shock2d};

// selectively pull from the asgard namespace
using P = asgard::default_precision;

using term_identity  = asgard::term_identity;
using term_volume   = asgard::term_volume<P>;
using term_div      = asgard::term_div<P>;
using term_grad     = asgard::term_grad<P>;
using term_chain     = asgard::term_chain;
using term_1d       = asgard::term_1d<P>;
using term_md       = asgard::term_md<P>;
using boundary_flux = asgard::boundary_flux<P>;
using term_robin = asgard::term_robin;
using separable_func = asgard::separable_func<P>;
using term_interp = asgard::term_interp<P>;

double constexpr PI = asgard::PI;

P step(P x, P cut, P s){
    return 1.0/(1.0 + exp(-(x-cut)/s));
}

class diffusion_params{
public:
    P w_min, w_max, u_max, height, s;
    // Value of the function

    P val(P const &u, P const &theta) const{
        if (s == 0){
            const P w = u * cos(theta);
            const bool condition = (u<u_max)*((w_min<w)&&(w<w_max));
            return condition? height : 0.0 ;
        }
        else{
            const P w = u * cos(theta);
            const P condition = step(u_max,u,s)*step(w,w_min,s)*step(w_max,w,s);
            return height * condition ;
        }
    }
};

asgard::pde_scheme<P> make(asgard::prog_opts  options)
{
    P Zi = options.file_required<P>("Zi");
    
    diffusion_params D{
        options.file_required<P>("cut_w_min"),
        options.file_required<P>("cut_w_max"),
        options.file_required<P>("cut_u"),
        options.file_required<P>("cut_height"),
        options.file_required<P>("cut_smooth")
    };
    
    P const u_max = options.file_required<P>("u_max");
    
    asgard::pde_domain<P> domain({{0.0, u_max}, {0.0, PI/2}});
    domain.set_names({"r", "theta"});
    
    // setting some default options
    // defaults are used only the corresponding values are missing from the command line
    options.default_degree = 2;
    options.default_start_levels = {5, };
    
    options.default_step_method = asgard::time_method::cn;
        
    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::none;
    
    options.default_isolver_tolerance  = 1.E-8;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 50;
    
    // create a pde from the given options and domain
    // we can read the variables using pde.options() and pde.domain() (both return const-refs)
    // the option entries may have been populated or updated with default values
    asgard::pde_scheme<P> pde(options, std::move(domain));
    
    // volume Jacobian in r
    auto dr = [](P r)-> P { return r * r; };
    // vector variants of the single dimensional volume Jacobian
    auto vec_dr = [=](std::vector<P> const &r, std::vector<P> &vol_r)
    -> void {
        for (size_t i = 0; i < r.size(); i++)
            vol_r[i] = dr(r[i]);
    };
    
    auto minus_vec_dr = [=](std::vector<P> const &r, std::vector<P> &vol_r)
    -> void {
        for (size_t i = 0; i < r.size(); i++)
            vol_r[i] = -dr(r[i]);
    };
    
    auto sqrt_dr = [=](std::vector<P> const &r, std::vector<P> &vol_r)
    -> void {
        for (size_t i = 0; i < r.size(); i++)
            vol_r[i] = r[i];
    };
    
    // volume Jacobian in theta
    auto dtheta = [](P theta)-> P { return sin(theta); };
    // vector variants of the single dimensional volume Jacobian
    auto vec_dtheta = [&](std::vector<P> const &theta, std::vector<P> &vol_theta)
    -> void {
        for (size_t i = 0; i < theta.size(); i++)
            vol_theta[i] = dtheta(theta[i]);
    };
    
    auto minus_vec_dtheta = [&](std::vector<P> const &theta, std::vector<P> &vol_theta)
    -> void {
        for (size_t i = 0; i < theta.size(); i++)
            vol_theta[i] = -dtheta(theta[i]);
    };
    
    // setting up the mass matrix
    pde.set_mass({term_volume{vec_dr}, term_volume{vec_dtheta}});
    
    // Collision Terms, u components
    {
        // Drift term: -1/u^2 d[u^2 *(-1/2u^2)f]/du
        pde += term_md({term_div(-0.5, asgard::flux_type::upwind),
            term_identity{}});
        
        // Diffusion term
        // s = 1/u^2 d/du(u^2 zeta q)
        term_1d div_grad(
                {term_div(-0.5, asgard::flux_type::upwind,
                          asgard::boundary_type::bothsides), // v^2 r = d/dv[ - 0.5 s]
                term_grad(sqrt_dr)} // v^2 s = v df/dv
        );

        div_grad.set_left_robin(0.5);
        
        // This needs improvement, the boundary condition is not df/dr + R f = 0
        //
        div_grad.set_right_robin(0.5);
        
        // s = 1/u^2 d/du[u^2 zeta df/du] = -1/2u^2 d/du(1/u df/du)
        pde += term_md({div_grad, term_identity{}});

        // Penalty term
        P const inv_dr = 1.0 / pde.cell_size(asgard::dimension_id{0});
        term_1d pen_r = asgard::term_penalty<P>{inv_dr, asgard::boundary_type::none};
        pde += term_md{{pen_r, term_identity{}}};
    }
    
    
    // Pitch angle diffusive term: (Z_i+1)/(4u^3 * sin(th)) d^2f/dth^2
    // Collision Terms, theta components
    {
        term_1d div_grad(
                {term_div(minus_vec_dtheta, asgard::flux_type::upwind,
                          asgard::boundary_type::bothsides), // sin(th) r = d/dth[-sin(th) s]
                term_grad(vec_dtheta)} // sin(th) s = sin(th) df/dth
        ); // r = -1/sin(th) d/dth(sin(th) df/dth)

        // 1/u^2 * (Zi+1)/4u * 1/sin(th) d/dth(sin(th) df/dth)
        // -(Zi+1)/4u^3sin(th) d/dth(sin(th) df/dth)
        P const fact = (Zi + 1.0)/4.0;
        term_1d Zi({term_volume{sqrt_dr}, term_volume{fact}});
        
        pde += term_md({Zi, div_grad});

        P const inv_dth = 1.0 / pde.cell_size(asgard::dimension_id{1});
        term_1d pen_th = asgard::term_penalty<P>{inv_dth, asgard::boundary_type::none};
        pde += term_md{{term_identity{},pen_th}};
    }
    
    // RF driver terms
    // div operator r
    term_md div_dr({term_div{minus_vec_dr, asgard::flux_type::upwind}, term_identity{}});
    // div operator theta
    term_md div_dtheta({term_volume{sqrt_dr}, term_div{minus_vec_dtheta, asgard::flux_type::upwind}});

    // grad operator r
    term_md grad_dr({term_grad{vec_dr}, term_identity{}});
    // grad operator theta
    term_md grad_dtheta({term_volume{sqrt_dr}, term_grad{vec_dtheta}});

    auto D_adapt = [=](P ,
                    asgard::vector2d<P> const &nodes,
                    std::vector<P> const & f,
                    std::vector<P> &vals)
    ->void{
        for (int64_t i = 0; i < nodes.num_strips(); i++){
            P const r = nodes[i][0];
            P const th = nodes[i][1];
            P vol = dr(r) * dtheta(th);
            
            vals[i] = D.val(r, th) * (1 - sin(2*th)) * f[i];
        }
    };
    pde.set_adapt_weight(D_adapt);
 
    
    // div_r D_rr grad_r term
    {

        auto D_rr = [=](P ,
                        asgard::vector2d<P> const &nodes,
                        std::vector<P> const & f,
                        std::vector<P> &vals)
        ->void{
            for (int64_t i = 0; i < nodes.num_strips(); i++){
                P const r = nodes[i][0];
                P const th = nodes[i][1];
                
                P vol = dr(r) * dtheta(th);
                vals[i] =  D.val(r, th) * pow(cos(th),2) * f[i];
            }
        };
        
            
        pde += term_md({div_dr, term_interp{D_rr}, grad_dr});
    }

    // div_r D_rtheta grad_theta term and,
    // div_theta D_rtheta grad_r term
    {
        auto D_rth = [=](P ,
                            asgard::vector2d<P> const &nodes,
                            std::vector<P> const & f,
                            std::vector<P> &vals)
        ->void{
            for (int64_t i = 0; i < nodes.num_strips(); i++){
                P const r = nodes[i][0];
                P const th = nodes[i][1];
                P vol = dr(r) * dtheta(th);
                vals[i] = - D.val(r, th) * cos(th) * sin(th) * f[i];
            }
        };
        pde += term_md({div_dr, term_interp{D_rth}, grad_dtheta});
        pde += term_md({div_dtheta, term_interp{D_rth}, grad_dr});
    }

    // div_theta D_thetatheta grad_theta term
    {
        auto D_thth = [=](P ,
                          asgard::vector2d<P> const &nodes,
                          std::vector<P> const & f,
                          std::vector<P> &vals)
        ->void{
            for (int64_t i = 0; i < nodes.num_strips(); i++){
                P const r = nodes[i][0];
                P const th = nodes[i][1];
                P vol = dr(r) * dtheta(th);
                
                vals[i] = D.val(r, th) * pow(sin(th),2) * f[i];
            }
            
        };
        pde += term_md({div_dtheta, term_interp{D_thth}, grad_dtheta});
    }

    // initial condition
    {
        auto initial_r = [=](std::vector<P> const &r,P ,std::vector<P> &fr) {
            for (size_t i = 0; i < r.size(); i++)
                fr[i] = dr(r[i]) * pow(2 * PI,-1.5) * exp(-r[i]*r[i]/2.0);
        };
        
        auto initial_th = [=](std::vector<P> const &th,P ,std::vector<P> &fth) {
            for (size_t i = 0; i < th.size(); i++)
                fth[i] = dtheta(th[i]);
        };

        pde.add_initial(separable_func({initial_r, initial_th}));
    }
    
    return pde;
}

int main(int argc, char** argv){
    // if MPI is enabled, call MPI_Init(), otherwise do nothing
    asgard::libasgard_runtime running_(argc, argv);
    
    // if double precision is available the P is double
    // otherwise P is float
    using P = asgard::default_precision;
    
    // parse the command-line inputs
    asgard::prog_opts options(argc, argv);

    // the discretization_manager takes in a pde and handles sparse-grid construction
    // separable and non-separable operators, holds the current state, etc.
    asgard::discretization_manager<P> disc(make(options),
                                           asgard::verbosity_level::high);

    disc.advance_time(); // integrate until num-steps or stop-time
    
    disc.progress_report();
    
    disc.save_final_snapshot(); // only if output filename is provided
    
    return 0;
};
