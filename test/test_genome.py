from agent import Genome

gen1 = Genome.random()
gen2 = Genome.random()
assert gen1 != gen2
gen3 = gen1.mutate()
assert gen1 != gen3
diff_list = [
    gen1.a - gen3.a,
    gen1.b - gen3.b,
    gen1.c - gen3.c,
    gen1.d - gen3.d,
    gen1.lr - gen3.lr,
]
print(diff_list)