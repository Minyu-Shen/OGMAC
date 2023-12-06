import matplotlib.pyplot as plt
from itertools import accumulate
from collections import defaultdict
import seaborn as sns

plt.rcParams['font.family'] = 'Latin Modern Roman'


loc_color = '#FBF4F9'
traje_color = '#408E91'
hold_color = '#E49393'

hold_color = '#FF0060'

# plot from 0 to plot_horiz seconds
plot_horiz = 3600.0 * 2.0
norm_time = 1.0  # in seconds

# cmap = plt.get_cmap('viridis')
# cmap = sns.color_palette("Spectral_r", as_cmap=True)
cmap = sns.color_palette("mako_r", as_cmap=True)
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)


def plot_time_space_diagram(buses, config):
    _, ax = plt.subplots()

    stop_xs = list(accumulate(config.link_lengs))
    stop_xs.insert(0, 0)
    stop_xs.pop()

    # set the right and top axis to be invisible
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Offset (km)', fontsize=12)

    # plot scheduled times
    # stop_sched_times = config.ln_info['stop_sched_times']
    # for stop, sched_times in stop_sched_times.items():
    #     x = stop_xs[stop] / 1000.0
    #     for sched_time in sched_times:
    #         if sched_time <= plot_horiz:
    #             ax.vlines(x=sched_time, ymin=x-0.1, ymax=x +
    #                       0.1, linewidth=1.0, color='r')

    # plotting horizontal stop location lines
    for x in stop_xs:
        ax.axhline(y=x/1000.0, color=loc_color, linestyle='-.',
                   dashes=(5, 2), linewidth=1.0)
    for bus in buses:
        # plot trajectory
        trip_xs = defaultdict(list)
        trip_ys = defaultdict(list)
        trip_occup_rates = defaultdict(list)
        for t, point in bus.traje.items():
            if t <= plot_horiz:
                trip_no = point['trip_no']
                trip_xs[trip_no].append(t/norm_time)
                trip_ys[trip_no].append(point["relat_x"] / 1000.0)
                occpu_rate = point['occup_rate']
                trip_occup_rates[trip_no].append(occpu_rate)
            else:
                break
        for trip, xs in trip_xs.items():
            ys = trip_ys[trip]
            occup_rates = trip_occup_rates[trip]
            for i in range(len(xs) - 1):
                x1, y1 = xs[i], ys[i]
                x2, y2 = xs[i + 1], ys[i + 1]
                # c = generate_color(occup_rates[i])
                # Map segment value to a color from the colormap
                c = cmap(occup_rates[i])
                # ax.plot([x1, x2], [y1, y2], color=c, linewidth=1.5)
                ax.plot([x1, x2], [y1, y2], c=c, linewidth=1.5)

            # ax.plot(xs, ys, color=traje_color, linewidth=1.5)

        # plot holding times
        hold_xs = defaultdict(list)
        hold_ys = {}
        for t, point in bus.traje.items():
            if t <= plot_horiz:
                trip_no = point['trip_no']
                if point['spot_type'] == 'hold':
                    hold_xs[(trip_no, point['spot_id'])].append(t/norm_time)
                    hold_ys[(trip_no, point['spot_id'])
                            ] = point['relat_x'] / 1000.0
            else:
                break

        for (trip_no, link_id), xs in hold_xs.items():
            start, end = min(xs), max(xs)
            y = hold_ys[(trip_no, link_id)]
            ax.hlines(y=y, xmin=start, xmax=end,
                      color=hold_color, linewidth=1.5)

    plt.colorbar(sm, label='Passenger occupancy rate', shrink=0.8)
    plt.show()
