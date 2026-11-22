import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evolve_xor import create_population, evolve
from rocket import simulate, brain_pilot, rocket_fitness



show_gens = [0, 5, 10, 20, 50, 100]
skip = 4
hold = 30
intro = 45
outro = 90
fps = 30

#same colors as the xor post image
bg = "#12172B"
panel = "#1B2238"
white = "#FFFFFF"
muted = "#A9B2CC"
green = "#23A26D"
red = "#E5484D"
gray = "#7C859E"
orange = "#F59E0B"


def fly(genome):
    pilot = lambda h , v , engine : brain_pilot(h , v , engine , genome)
    fuel , h , v , history = simulate(pilot , record = True)
    return fuel , h , v , history


def result(h , v):
    if h > 0:
        return "away" , gray
    if abs(v) < 2:
        return "landed" , green
    return "crashed" , red


def count_landings(population):
    landed = 0
    for genome in population:
        fuel , h , v , history = fly(genome)
        if result(h , v)[0] == "landed":
            landed += 1
    return landed


def train_generations():
    population = create_population(50)

    #save the whole population for the generations we show
    #and count soft landings for every generation for the chart
    saved = {0: population}
    landings = [count_landings(population)]

    for gen in range(1, show_gens[-1] + 1):
        population = evolve(population, rocket_fitness, 1)
        landings.append(count_landings(population))
        if gen in show_gens:
            saved[gen] = population

    return saved , landings


if __name__ == "__main__":
    saved , landings = train_generations()

    #fly every rocket of every saved generation, best rockets on the left
    all_flights = []
    for gen in show_gens:
        population = sorted(saved[gen], key=rocket_fitness)
        all_flights.append([fly(genome) for genome in population])

    #each frame is (which generation, which step, is it the end card)
    frames = []
    for g in range(len(show_gens)):
        longest = max(len(flight[3]) for flight in all_flights[g])
        if g == 0:
            for _ in range(intro):
                frames.append((g , 0 , False))
        for step in range(0, longest, skip):
            frames.append((g , step , False))
        for _ in range(hold):
            frames.append((g , longest - 1 , False))
    for _ in range(outro):
        frames.append((len(show_gens) - 1 , 10**6 , True))

    n = len(all_flights[0])

    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=bg)

    #headline
    fig.text(0.04, 0.95, f"{n} neural networks learning to land a rocket",
             color=white, fontsize=26, weight="bold", va="center")
    fig.text(0.04, 0.915, "No backprop. Each generation the best brains survive, breed and mutate.",
             color=muted, fontsize=15, va="center")

    gen_text = fig.text(0.04, 0.868, "", color=white, fontsize=30, weight="bold", va="center")
    landed_text = fig.text(0.62, 0.868, "", color=green, fontsize=17, weight="bold", ha="right", va="center")
    crashed_text = fig.text(0.79, 0.868, "", color=red, fontsize=17, weight="bold", ha="right", va="center")
    away_text = fig.text(0.96, 0.868, "", color=gray, fontsize=17, weight="bold", ha="right", va="center")

    #rockets
    ax = fig.add_axes([0.04, 0.34, 0.92, 0.495], facecolor=panel)
    ax.set_xlim(0, n + 1)
    ax.set_ylim(-5, 125)
    ax.axhspan(-5, 0, color="#3A4566")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 0.97, "best on the left", transform=ax.transAxes,
            color=muted, fontsize=11, va="top")

    end_text = ax.text(0.5, 0.6, "", transform=ax.transAxes, color=white,
                       fontsize=30, weight="bold", ha="center", va="center")

    rockets = []
    flames = []

    for i in range(n):
        rocket, = ax.plot([], [], marker="^", markersize=12, color=white, linestyle="none")
        flame, = ax.plot([], [], marker="v", markersize=8, color=orange, linestyle="none")
        rockets.append(rocket)
        flames.append(flame)

    #learning curve
    fig.text(0.10, 0.285, f"Soft landings per generation (out of {n})",
             color=muted, fontsize=14, va="center")
    chart = fig.add_axes([0.10, 0.07, 0.86, 0.19], facecolor=bg)
    chart.set_xlim(0, show_gens[-1])
    chart.set_ylim(0, n)
    chart.tick_params(colors=muted, labelsize=11)
    for spine in chart.spines.values():
        spine.set_color(muted)
    chart.set_xlabel("Generation", color=muted, fontsize=12)

    curve, = chart.plot([], [], color=green, linewidth=3)
    dot, = chart.plot([], [], marker="o", markersize=10, color=green, linestyle="none")


    def update(frame):
        g , step , end_card = frames[frame]
        gen = show_gens[g]
        counts = {"landed": 0, "crashed": 0, "away": 0}

        for i in range(n):
            fuel , h_end , v_end , history = all_flights[g][i]

            #keep a finished rocket on its last position
            s = min(step , len(history) - 1)
            h , engine = history[s]
            done = s == len(history) - 1

            #rockets that fly too high stay stuck at the top so we can still see them
            x = i + 1
            y = min(max(h , 0) , 118) + 2
            rockets[i].set_data([x], [y])

            if engine and not done:
                flames[i].set_data([x], [y - 3.5])
            else:
                flames[i].set_data([], [])

            if done:
                outcome , color = result(h_end , v_end)
                rockets[i].set_color(color)
                counts[outcome] += 1
            elif h > 118:
                rockets[i].set_color(gray)
            else:
                rockets[i].set_color(white)

        gen_text.set_text(f"Generation {gen}")
        landed_text.set_text(f"{counts['landed']} landed")
        crashed_text.set_text(f"{counts['crashed']} crashed")
        away_text.set_text(f"{counts['away']} flew away")

        #draw the curve up to the generation we are showing
        curve.set_data(range(gen + 1), landings[:gen + 1])
        dot.set_data([gen], [landings[gen]])

        if end_card:
            end_text.set_text(f"Generation 0: {landings[0]} soft landings\n"
                              f"Generation {gen}: {landings[gen]}")
        else:
            end_text.set_text("")

        return rockets + flames


    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        repeat=False
    )

    #mp4 for linkedin, smaller gif for the github readme
    animation.save("rockets.mp4", writer="ffmpeg", fps=fps)
    animation.save("rockets.gif", writer="pillow", fps=fps, dpi=60)

    print("\nResults")
    for gen , flights in zip(show_gens , all_flights):
        outcomes = [result(flight[1] , flight[2])[0] for flight in flights]
        print(f"Gen {gen}: landed {outcomes.count('landed')}, crashed {outcomes.count('crashed')}, did not land {outcomes.count('away')}")
    print("landings per generation:", landings)
    print("saved rockets.mp4 and rockets.gif")
