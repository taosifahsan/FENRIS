#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

// LHCD 1-D quasilinear Fokker-Planck solver on v (Cartesian, no mass).
//
// Solves  df/dt = d/dv[ A(v) df/dv + B(v) f ],  A = D_RF + I^3/2, B = -I^2,
//   with I(v) a regularized 1/v and D_RF a window in v.
// Boundaries: bothsides on both divs -> zero total flux at both walls.
#include "asgard.hpp"

// User-editable f0(v); lives in input_data/ because it is an input.
#include "initial_condition.hpp"

using P = asgard::default_precision;

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

// RF diffusion strength: a smoothed window [min, max] of height `height`.
class diffusion_params{
public:
    P min, max, sharpness, height;
    P val(P const &v) const{
        return height * step(v - min) * step(max - v);
    }
    P step(P const &v) const{
        return sharpness != 0.0 ? 1/(1 + exp(-v/sharpness)) : v > 0;
    }
};

inline P sign(P const &v) {
    return v > 0 ? 1 : - 1;
}

// Regularized 1/v: softened near zero, clamped outside [inv_min, inv_max].
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

pde_scheme make_pde(asgard::prog_opts options)
{
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

    pde_domain domain({{
        options.file_required<P>("domain_min"),
        options.file_required<P>("domain_max")
    }});

    P const dx = domain.min_cell_size(options.max_level());

    options.default_step_method = asgard::time_method::cn;
    options.default_solver = asgard::solver_method::gmres;
    options.default_precon = asgard::precon_method::none;

    options.default_isolver_tolerance  = 1.E-12;
    options.default_isolver_iterations = 1000;
    options.default_isolver_inner_iterations = 400;

    pde_scheme pde(options, std::move(domain));

    // Adaptivity weight: refine where RF diffusion is strong and f is present.
    auto D_adapt = [=](P, asgard::vector2d<P> const &nodes,
                       std::vector<P> const &f, std::vector<P> &value) {
        for (std::int64_t i = 0; i < nodes.num_strips(); ++i) {
            P const v = nodes[i][0];
            value[i] = std::abs(D.val(v)) * std::abs(f[i]);
        }
    };
    pde.set_adapt_weight(D_adapt);

    // Diffusion: -d/dv(A(v) df/dv).  A is separable in the one coordinate,
    // so it sits inside the div rather than going through term_interp.
    {
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

    // Drag: -d/dv(B(v) f).
    {
        auto B = [=](const vector &v, vector &func) {
            for (size_t i = 0; i < v.size(); ++i)
                func[i] =  -sign(v[i]) * pow(I.val(v[i]), 2);
        };

        term_md Bdiv{term_div(B, boundary_type::bothsides),};
        pde += term_md({Bdiv});

    }

    // Initial condition from input_data/initial_condition.hpp, normalized by
    // composite Simpson so integral f0 dv = 1 over the actual domain.
    {
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

    return pde;
}

int main(int argc, char **argv)
{
    asgard::prog_opts options(argc, argv);
    disc_manager disc(make_pde(options), verbosity_level::high);

    // Movie snapshots: one .h5 per frame under movie_dir.  movie_frames
    // spreads roughly that many over the run; movie_stride overrides it;
    // both zero disables snapshots.
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
