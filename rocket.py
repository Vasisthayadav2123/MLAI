from two_layer import forward_pass
from evolve_xor import create_population,evolve 



def simulate(pilot , record = False):
    t = 0
    dt = 0.1
    h= 100
    v=0
    step = 0
    engine = False
    fuel = 0
    history = [(h , engine)]
    while h>0:
        engine = pilot(h, v, engine)
        v =  v + (engine * 15 - 9.81) * dt
        h = h + v * dt
        t = t + dt
        step = step + 1
        if engine:
            fuel = fuel + 1
        history.append((h , engine))
        if step > 300:
            break
    #only the animation needs the history
    if record:
        return fuel , h , v , history
    return fuel , h , v

def brain_pilot(h , v , engine , genome):
    h_scaled = h / 100
    v_scaled = v / 30

    #netwokr predict
    a1, a2, p = forward_pass([h_scaled , v_scaled] , genome)
    return p> 0.5

def my_pilot(h , v , engine):
    if h < 4 or v < -25 and not engine:
        engine = True
    if v > -1.5 and engine:
        engine = False
    return engine




def rocket_fitness(genome):
    pilot = lambda h , v , engine :brain_pilot(h , v , engine , genome)
    fuel, h, v = simulate(pilot)

    score = abs(v) + 0.01 *fuel 

    if h>0:
        score += 200 + 0.1 * h 

    return score

if __name__ == "__main__":
    population = create_population(50)

    for generation in range(100):
        population = evolve(
            population,
            rocket_fitness,
            1
        )

        if generation % 10 == 0:
            best = min(population, key=rocket_fitness)
            print(
                f"Generation {generation}: "
                f"Best score = {rocket_fitness(best):.2f}"
            )

    winner = min(population, key=rocket_fitness)

    pilot = lambda h, v, engine: brain_pilot(
        h, v, engine, winner
    )

    fuel, h, v = simulate(pilot)

    print("\nWINNING ROCKET")
    print("Score:", rocket_fitness(winner))
    print("Fuel:", fuel)
    print("Height:", h)
    print("Landing speed:", abs(v))

    if h <= 0 and abs(rocket_fitness(winner)) < 1.80:
        print("SUCCESS! Beat the 1.80 benchmark!")
    else:
        print("Benchmark not beaten yet.")