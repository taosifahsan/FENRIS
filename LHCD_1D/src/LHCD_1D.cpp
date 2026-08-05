#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "asgard.hpp"

// The user-editable initial condition f0(v): lives in input_data/ because it
// is an input, even though it is code.  See the header itself for the contract.
#include "initial_condition.hpp"

// Use ASGarD's default precision (typically double)
using P = asgard::default_precision;

// Convenient aliases for common ASGarD types
using disc_manager = asgard::discretization_manager<P>;
using pde_scheme = asgard::pde_scheme<P>;
using term_1d = asgard::term_1d<P>;
using term_md = asgard::term_md<P>;
using term_div = asgard::term_div<P>;
using term_volume = asgard::term_volume<P>;
using term_interp = asgard::term_interp<P>;
using term_grad = asgard::term_grad<P>;
using term_penalty = asgard::term_penalty<P>;
using pde_domain = asgard::pde_domain<P>;
using separable_func = asgard::separable_func<P>;
using vector = std::vector<P>;
using asgard::verbosity_level;
using asgard::flux_type;
using asgard::boundary_type;
using asgard::right_boundary_flux;
using asgard::left_boundary_flux;
// using asgard::ignores_time;

// ───────────────────────────────────────
// Utility Class for Diffusion Coefficient
// ───────────────────────────────────────

class diffusion_params{
public:
    P min, max, sharpness, height;
    // Value of the function
    P val(P const &v) const{
        return height * step(v - min) * step(max - v);
    }
    // step function
    P step(P const &v) const{
        return sharpness != 0.0 ? 1/(1 + exp(-v/sharpness)) : v > 0;
    }
};

// sign function
inline P sign(P const &v) {
    return v > 0 ? 1 : - 1;
}

// Regularizer of 1/v
class inverse_params{
public:
    P eps, inv_min, inv_max;
    // value of the function
    P val(P const &v) const{
        if ((inv_min < v)&&(v < inv_max)) {
            return sign(v)/sqrt(v*v+eps*eps);
        }
        else if (v <= inv_min){
            return 1/inv_min;
        }
        else{
            return 1/inv_max;
        }
    }
};

// ──────────────────────────────
// PDE Constuction
// ──────────────────────────────

pde_scheme make_pde(asgard::prog_opts options)
{
    // Read parameters
    diffusion_params D{
        options.file_required<P>("cut_min"),
        options.file_required<P>("cut_max"),
        options.file_required<P>("cut_sharpness"),
        options.file_required<P>("cut_height")
    };
    inverse_params I{
        options.file_required<P>("epsilon"),
        options.file_required<P>("inverse_min"),
        options.file_required<P>("inverse_max")
    };

    options.default_degree = 1;
    options.default_start_levels = {8, };

    // Setting domain conditional on side
    pde_domain domain({{
        options.file_required<P>("domain_min"),
        options.file_required<P>("domain_max")
    }});

    P const dx = domain.min_cell_size(options.max_level());

    // Set solver method
    options.default_step_method = asgard::time_method::cn;
    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::none;

    options.default_isolver_tolerance  = 1.E-12;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 400;

    // Declaring the PDE
    pde_scheme pde(options, std::move(domain));

    // Adaptive-grid weight, mirroring LHCD_2D's D_adapt exactly: refine
    // where the RF diffusion coefficient is large and the solution is
    // present.
    auto D_adapt = [=](P, asgard::vector2d<P> const &nodes,
                       std::vector<P> const &f, std::vector<P> &value) {
        for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
            P const v = nodes[i][0];
            value[i] = std::abs(D.val(v)) * std::abs(f[i]);
        }
    };
    pde.set_adapt_weight(D_adapt);

    {// First term -d/dv (A(v) * df/dv)

        // A(v) is a plain function of the single coordinate, so it belongs
        // directly inside the div as a separable coefficient -- the same
        // div-grad chain construction the ICRF solvers use.  (The original
        // routed it through term_interp, which exists for coefficients that
        // cannot be written separably; in 1-D nothing qualifies, and the
        // interpolation pass only added cost and interpolation error.)
        auto negA = [=](const vector &v, vector &func) {
            for (size_t i = 0; i < v.size(); ++i)
                func[i] = -sign(v[i])
                        * (D.val(v[i]) + 0.5 * pow(I.val(v[i]), 3));
        };

        term_1d div_grad({
                term_div(negA, boundary_type::bothsides), // d/dv[ -A(v) s ]
                term_grad(1) // s = df/dv
        });
        pde += term_md({div_grad});

        pde += term_md{ term_penalty{1/dx}, };
    }

    {// Second term -d/dv (B * f)

        auto B = [=](const vector &v, vector &func) {
            for (size_t i = 0; i < v.size(); ++i)
                func[i] =  -sign(v[i]) * pow(I.val(v[i]), 2);
        };
        
        // bothsides: the advective flux is fixed to zero at both walls.
        // Previously this bracket was left free with no cancelling Robin --
        // a latent leak, benign only because f(+-v_max) is tiny; the same
        // defect class caused the 2-D projects' particle drift.
        term_md Bdiv{term_div(B, boundary_type::bothsides),};
        pde += term_md({Bdiv});

    }

    // initial condition f(0,v) = f_0(v), user-supplied as the lambda in
    // input_data/initial_condition.hpp (a compile input: editing it triggers
    // a rebuild and a re-solve through the normal CMake staleness rules).
    {
        // Normalize numerically so integral f_0 dv = 1 over the actual
        // domain, whatever shape the lambda returns (composite Simpson;
        // microseconds).  Diagnostics divide by N(0) and the plotters
        // renormalize, so this changes no downstream figure -- it only fixes
        // the absolute scale that the old hard-coded exp(-v^2) left at
        // sqrt(pi).
        P const v_lo = options.file_required<P>("domain_min");
        P const v_hi = options.file_required<P>("domain_max");
        P norm = 0;
        {
            int const n = 1 << 16;  // intervals; even, as Simpson requires
            P const h = (v_hi - v_lo) / n;
            for (int i = 0; i <= n; i++) {
                P const v = v_lo + i * h;
                P const w = (i == 0 || i == n) ? 1 : (i % 2 ? 4 : 2);
                norm += w * initial_f0(v);
            }
            norm *= h / 3;
            if (!(norm > 0) || !std::isfinite(norm))
                throw std::runtime_error(
                    "initial_condition.hpp: integral of f0(v) over the "
                    "domain must be positive and finite");
        }
        auto init = [norm](vector const &v, P , vector &func){
            for (size_t i = 0; i < v.size(); i++)
                func[i] = initial_f0(v[i]) / norm;
        };

        pde.add_initial(separable_func({init}));
    }

    // df/dt - d/dv (A(v) * df/dv + B(v) * f) = 0
    return pde;
}

// ──────────────────────────────
// Main Execution:
// ──────────────────────────────

int main(int argc, char **argv)
{
    // Parse options from CLI or config file
    asgard::prog_opts options(argc, argv);

    // Construct PDE and discretization manager
    disc_manager disc(make_pde(options), verbosity_level::high);

    // Movie snapshots: same deck keys and file layout as the 2D solvers --
    // separate per-frame .h5 files under movie_dir, replacing the old scheme
    // of aux fields accumulated inside one file.  movie_frames spreads
    // roughly that many snapshots over the run (including initial and final);
    // movie_stride overrides it when positive; both zero disables snapshots.
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
        std::cerr << "LHCD_1D: time advancement stopped before the requested "
                     "final time.\n";
        return 2;
    }

    return EXIT_SUCCESS;
}
