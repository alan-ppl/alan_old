from ammpis import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

import torch as t

from torch.distributions import Normal, Laplace, Gumbel

t.manual_seed(0)
t.cuda.manual_seed(0)

if __name__ == "__main__":
    num_runs = 1

    num_iters = 100 # mcmc and lang seem to require about 1000 iterations to converge
    
    num_latents = [10, 100, 1000]
    

    posteriors = {"Normal": Normal, "Laplace": Laplace, "Gumbel": Gumbel}
        
    
    errors = {"ammp_is_post_q":   t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_post_avg": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_uniform_dt_post_q":   t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_uniform_dt_post_avg": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_no_inner_loop_post_q":   t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_no_inner_loop_post_avg": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_weight_all_post_q":   t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_weight_all_post_avg": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "mcmc_post":        t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "lang_post":        t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2)}



    entropies = {"ammp_is": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
            "ammp_is_uniform_dt": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
            "ammp_is_no_inner_loop": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
            "ammp_is_weight_all": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
            "mcmc":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
            "lang":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1)}
    
    weighted_ltots = {"ammp_is": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
        "ammp_is_uniform_dt": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
        "ammp_is_no_inner_loop": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
        "ammp_is_weight_all": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
        "mcmc":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
        "lang":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1)}
    

    times = {"ammp_is": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "ammp_is_uniform_dt": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "ammp_is_no_inner_loop": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "ammp_is_weight_all": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "mcmc":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "lang":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1)}
    
    for run in range(num_runs):
        for post in posteriors:
            for n in num_latents:
                print(f"run: {run}, post: {post}, n: {n}", end="\t")
            
                init = t.tensor([0.0,1.0], dtype=t.float64).repeat((n,1))

                # priors on posterior parameters
                loc = Normal(0,100).sample((n,1)).float()
                scale = Normal(0,0.1).sample((n,1)).exp().float() + 5
                # do log uniform instead of uniform to avoid numerical issues

                post_dist = posteriors[post](loc.squeeze(), scale.squeeze())
                true_post = t.cat([post_dist.mean.unsqueeze(1), post_dist.stddev.unsqueeze(1)], dim=1)

                for fn in [ammp_is, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_weight_all]:

                    name = fn.__name__
                    # ammpis
                    m_q, m_avg, l_tot, weights, ents, fn_times = fn(num_iters, post_dist, init, 0.4, 100)
                    post_q = [fit_approx_post(m) for m in m_q]
                    post_avg = [fit_approx_post(m) for m in m_avg]

                    # calculate errors
                    for i in range(num_iters+1):
                        errors[f"{name}_post_q"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - post_q[i]).abs().mean(0)
                        errors[f"{name}_post_avg"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - post_avg[i]).abs().mean(0)

                    times[f"{name}"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = fn_times
                    entropies[f"{name}"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = t.tensor(ents)

                    weighted_ltots[f"{name}"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = t.tensor([lt + wt for lt, wt in zip(l_tot, weights)])


                # mcmc
                burn_in = num_iters//10
                m_mcmc, mcmc_acceptance_rate, mcmc_times, mcmc_samples = mcmc(num_iters, post_dist, init, 2.4*post_dist.stddev.unsqueeze(1), burn_in=burn_in)
                mcmc_post = [fit_approx_post(m) for m in m_mcmc]

                # langevin (mala)
                m_lang, lang_acceptance_rate, lang_times, lang_samples = lang(num_iters, post_dist, init, 2.4*post_dist.stddev.unsqueeze(1), burn_in=burn_in)
                lang_post = [fit_approx_post(m) for m in m_lang]

                print(f"Acceptance rates: mcmc={mcmc_acceptance_rate},\t lang={lang_acceptance_rate}")

                # calculate errors
                for i in range(num_iters+1):
                    errors["mcmc_post"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - mcmc_post[i]).abs().mean(0)
                    errors["lang_post"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - lang_post[i]).abs().mean(0)

                times["mcmc"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = mcmc_times
                times["lang"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = lang_times

                # plot the pdfs of the final approximating distributions
                # if run == 0 and not (post == "Gumbel" and n > 10):
                # if run == 0 and post == "Gumbel" and n != 1000:
                #     fig, ax = plt.subplots(1,1, figsize=(5.5, 4.0))
                #     # breakpoint()
                #     df = pd.DataFrame({"true": post_dist.sample((num_iters*10,)).numpy()[:,0],
                #                     #    "ammp_is": Normal(ammp_is_post_avg[-1][:,0], ammp_is_post_avg[-1][:,1]).sample((num_iters*10,)).numpy()[:,0],
                #                        "mcmc": mcmc_samples[burn_in:,0].repeat(10),
                #                        "lang": lang_samples[burn_in:,0].repeat(10)})
                #                         # "mcmc": Normal(mcmc_post[-1][:,0], mcmc_post[-1][:,1]).sample((num_iters*10,)).numpy()[:,0],
                #                         # "lang": Normal(lang_post[-1][:,0], lang_post[-1][:,1] + 1e-10).sample((num_iters*10,)).numpy()[:,0]})

                #     df.plot.kde(ax=ax, legend=True, ind=None)

                #     ax.set_title(f"n={n}, {post}({loc[0,0].item():.3f}, {scale[0,0].item():.3f}) posterior")
                #     plt.savefig(f"figures/{post}_n{n}_run{run}.png")
                #     plt.close()


    # take mean over runs
    for fn in [ammp_is, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_weight_all]:
        name = fn.__name__
        errors[f"{name}_post_q"] = errors[f"{name}_post_q"].mean(0)
        errors[f"{name}_post_avg"] = errors[f"{name}_post_avg"].mean(0)
        times[f"{name}"] = times[f"{name}"].mean(0)
        entropies[f"{name}"] = entropies[f"{name}"].mean(0)
        weighted_ltots[f"{name}"] = weighted_ltots[f"{name}"].mean(0)


    errors["mcmc_post"] = errors["mcmc_post"].mean(0)
    errors["lang_post"] = errors["lang_post"].mean(0)


    times["mcmc"] = times["mcmc"].mean(0)
    times["lang"] = times["lang"].mean(0)




    for x_axis in ["iterations", "time"]:
        for post in posteriors:
            # subplot for each num_latents and each element of the error tensor (mean, scale)
            fig, axs = plt.subplots(len(num_latents), 4, figsize=(16,16), sharex=(x_axis=="iterations"))
            fig.suptitle(post, x=0.17, y=0.95, weight="bold")

            for fn in [ammp_is, ammp_is_uniform_dt, ammp_is_no_inner_loop, ammp_is_weight_all, mcmc, lang]:
                name = fn.__name__
                for i in range(len(num_latents)):
                    if x_axis == "iterations":
                        xs = t.arange(num_iters+1)
                    elif x_axis == "time":
                        xs = times[f"{name}"][list(posteriors.keys()).index(post), i, :]

                    if name in ['mcmc', 'lang']:
                        axs[i, 0].plot(xs, errors[f"{name}_post"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label=f"{name}")
                        axs[i, 1].plot(xs, errors[f"{name}_post"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label=f"{name}")
                    else:
                        axs[i, 0].plot(xs, errors[f"{name}_post_q"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label=f"{name}_Qt")
                        axs[i, 0].plot(xs, errors[f"{name}_post_avg"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label=f"{name}_av")

                        axs[i, 1].plot(xs, errors[f"{name}_post_q"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label=f"{name}_Qt")
                        axs[i, 1].plot(xs, errors[f"{name}_post_avg"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label=f"{name}_av")

                        axs[i, 2].plot(xs[2:], entropies[f"{name}"][list(posteriors.keys()).index(post), i, :].numpy()[2:], label=f"{name}")
                        axs[i, 3].plot(xs[2:], weighted_ltots[f"{name}"][list(posteriors.keys()).index(post), i, :].numpy()[2:], label=f"{name}")
                
                axs[0, 0].set_title(f"Error in mean")
                axs[0, 1].set_title("Error in variance")
                axs[0, 2].set_title("Entropy")
                axs[0, 3].set_title("Weighted l_tot")


                # add x-axis labels
                for ax in axs[-1,:]:
                    ax.set_xlabel("Iterations" if x_axis == "iterations" else "Time (s)")

                # for each row, add a label to the y-axis containing the num_latents
                for ax, row in zip(axs[:,0], num_latents):
                    ax.set_ylabel(f"n={row}")


            # add horizontal legend along top of plot
            axs[0,1].legend(loc='lower center', bbox_to_anchor=(0.27, 1.25), shadow=False, ncol=2)

            plt.savefig(f"figures/{post}{'_vs_time' if x_axis == 'time' else ''}.png")
            plt.close()


# STEP 2 would be to try non-Gaussian proposals/approximate posteriors?
