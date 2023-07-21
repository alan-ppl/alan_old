from ammpis import *
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import math
import torch as t

from torch.distributions import Normal, Laplace, Gumbel

t.manual_seed(0)
t.cuda.manual_seed(0)

if __name__ == "__main__":
    num_runs = 5

    num_iters = 1000 # mcmc seems to require about 1000 iterations to converge
    
    num_latents = [10, 100, 1000, 10000]
    
    posteriors = {"Normal": Normal, "Laplace": Laplace, "Gumbel": Gumbel}
        
    errors = {"ammp_is_post_q":   t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "ammp_is_post_avg": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "mcmc_post":        t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2),
              "lang_post":        t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1, 2)}

    times = {"ammp_is": t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "mcmc":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1),
             "lang":    t.zeros(num_runs, len(posteriors), len(num_latents), num_iters+1)}
    
    for run in range(num_runs):
        for post in posteriors:
            for n in num_latents:
                print(f"run: {run}, post: {post}, n: {n}")
            
                init = t.tensor([0.0,1.0], dtype=t.float64).repeat((n,1))

                # priors on posterior parameters
                loc = Normal(0,100).sample((n,1)).float()
                scale = Normal(0,0.1).sample((n,1)).exp().float()

                post_dist = posteriors[post](loc.squeeze(), scale.squeeze())
                true_post = t.cat([post_dist.mean.unsqueeze(1), post_dist.stddev.unsqueeze(1)], dim=1)

                # ammpis
                m_q, m_avg, l_tot, ammp_is_times = ammp_is(num_iters, post_dist, init, 0.4, 100)
                ammp_is_post_q = [fit_approx_post(m) for m in m_q]
                ammp_is_post_avg = [fit_approx_post(m) for m in m_avg]

                # mcmc
                m_mcmc, mcmc_acceptance_rate, mcmc_times = mcmc(num_iters, post_dist, init, 2.4*post_dist.stddev.unsqueeze(1), burn_in=0)#num_iters//10)
                mcmc_post = [fit_approx_post(m) for m in m_mcmc]

                # langevin (mala)
                m_lang, lang_acceptance_rate, lang_times = am(num_iters, post_dist, init, 2.4*post_dist.stddev.unsqueeze(1), burn_in=0)#num_iters//10)
                lang_post = [fit_approx_post(m) for m in m_lang]

                # calculate errors
                for i in range(num_iters+1):
                    errors["ammp_is_post_q"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - ammp_is_post_q[i]).abs().mean(0)
                    errors["ammp_is_post_avg"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - ammp_is_post_avg[i]).abs().mean(0)
                    errors["mcmc_post"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - mcmc_post[i]).abs().mean(0)
                    errors["lang_post"][run, list(posteriors.keys()).index(post), num_latents.index(n), i, :] = (true_post - lang_post[i]).abs().mean(0)


                times["ammp_is"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = ammp_is_times
                times["mcmc"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = mcmc_times
                times["lang"][run, list(posteriors.keys()).index(post), num_latents.index(n), :] = lang_times

    # take mean over runs
    errors["ammp_is_post_q"] = errors["ammp_is_post_q"].mean(0)
    errors["ammp_is_post_avg"] = errors["ammp_is_post_avg"].mean(0)
    errors["mcmc_post"] = errors["mcmc_post"].mean(0)
    errors["lang_post"] = errors["lang_post"].mean(0)

    times["ammp_is"] = times["ammp_is"].mean(0)
    times["mcmc"] = times["mcmc"].mean(0)
    times["lang"] = times["lang"].mean(0)

    for x_axis in ["iterations", "time"]:
        for post in posteriors:
            # subplot for each num_latents and each element of the error tensor (mean, scale)
            fig, axs = plt.subplots(len(num_latents), 2, figsize=(8,8), sharex=(x_axis=="iterations"))
            fig.suptitle(post, x=0.17, y=0.95, weight="bold")

            for i in range(len(num_latents)):
                if x_axis == "iterations":
                    ampp_is_xs = t.arange(num_iters+1)
                    mcmc_xs = t.arange(num_iters+1)
                elif x_axis == "time":
                    ampp_is_xs = times["ammp_is"][list(posteriors.keys()).index(post), i, :]
                    mcmc_xs = times["mcmc"][list(posteriors.keys()).index(post), i, :]

                axs[i, 0].plot(ampp_is_xs, errors["ammp_is_post_q"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label="ammp_is_Qt")
                axs[i, 0].plot(ampp_is_xs, errors["ammp_is_post_avg"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label="ammp_is_av")
                axs[i, 0].plot(mcmc_xs, errors["mcmc_post"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label="mcmc")
                axs[i, 0].plot(mcmc_xs, errors["lang_post"][list(posteriors.keys()).index(post), i, :, 0].numpy(), label="lang")
                # axs[i, 0].set_ylim([0, 1])

                axs[i, 1].plot(ampp_is_xs, errors["ammp_is_post_q"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label="ammp_is_Qt")
                axs[i, 1].plot(ampp_is_xs, errors["ammp_is_post_avg"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label="ammp_is_avg")
                axs[i, 1].plot(mcmc_xs, errors["mcmc_post"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label="mcmc")
                axs[i, 1].plot(mcmc_xs, errors["lang_post"][list(posteriors.keys()).index(post), i, :, 1].numpy(), label="langevin")
                # axs[i, 1].set_ylim([0, 1])
            
            axs[0, 0].set_title(f"Error in mean")
            axs[0, 1].set_title("Error in variance")


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
