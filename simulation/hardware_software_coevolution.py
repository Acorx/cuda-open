"""
CUDA Open - Co-Evolution Hardware/Software

Évolue SIMULTANÉMENT:
1. L'architecture matérielle (nombre d'unités, mémoire, etc.)
2. Les algorithmes logiciels (stratégies d'entraînement, kernels)

C'est le Saint Graal: découvrir la paire hardware/software optimale
que personne n'aurait imaginée séparément.

Usage:
    python3 simulation/hardware_software_coevolution.py
"""

import numpy as np
import time
import json
from typing import Dict, Tuple
from copy import deepcopy
from dataclasses import dataclass, field


# ============================================================================
# GENOME COMPOSITE (Hardware + Software)
# ============================================================================

@dataclass
class HardwareGenome:
    """Architecture matérielle."""
    num_compute_units: int = 1024
    unit_clock_ghz: float = 1.5
    memory_bandwidth_gbs: float = 1000
    sram_size_kb: int = 256
    num_registers: int = 256
    data_path_width: int = 32  # bits
    pipeline_depth: int = 4
    support_ternary: bool = True
    support_sparse: bool = False
    interconnect_topology: str = "mesh"  # mesh, ring, crossbar

    def to_vector(self):
        topo_map = {"mesh": 0, "ring": 1, "crossbar": 2}
        return np.array([
            np.log2(max(64, self.num_compute_units)),
            self.unit_clock_ghz,
            np.log2(max(100, self.memory_bandwidth_gbs)),
            np.log2(max(64, self.sram_size_kb)),
            np.log2(max(64, self.num_registers)),
            np.log2(max(8, self.data_path_width)),
            self.pipeline_depth,
            int(self.support_ternary),
            int(self.support_sparse),
            topo_map.get(self.interconnect_topology, 0)
        ])

    @classmethod
    def from_vector(cls, v):
        topo_r = {0: "mesh", 1: "ring", 2: "crossbar"}
        return cls(
            num_compute_units=int(2**max(6, v[0])),
            unit_clock_ghz=max(0.5, min(5.0, v[1])),
            memory_bandwidth_gbs=int(2**max(6, v[2])),
            sram_size_kb=int(2**max(6, v[3])),
            num_registers=int(2**max(6, v[4])),
            data_path_width=int(2**max(3, min(6, v[5]))),
            pipeline_depth=int(max(1, min(16, v[6]))),
            support_ternary=bool(v[7] > 0.5),
            support_sparse=bool(v[8] > 0.5),
            interconnect_topology=topo_r.get(int(v[9]) % 3, "mesh")
        )


@dataclass
class SoftwareGenome:
    """Stratégie logicielle."""
    quantization_bits: int = 2
    group_size: int = 128
    fused_kernel_depth: int = 3  # Nombre d'opérations fusionnées
    scheduling_policy: str = "dynamic"  # static, dynamic, learned
    memory_tiling: bool = True
    prefetch_distance: int = 4
    use_tensor_cores: bool = True
    gradient_accumulation: int = 1
    activation_checkpointing: bool = False

    def to_vector(self):
        sched_map = {"static": 0, "dynamic": 1, "learned": 2}
        return np.array([
            np.log2(max(1, self.quantization_bits)),
            np.log2(max(32, self.group_size)),
            self.fused_kernel_depth,
            sched_map.get(self.scheduling_policy, 1),
            int(self.memory_tiling),
            self.prefetch_distance,
            int(self.use_tensor_cores),
            np.log2(max(1, self.gradient_accumulation)),
            int(self.activation_checkpointing)
        ])

    @classmethod
    def from_vector(cls, v):
        sched_r = {0: "static", 1: "dynamic", 2: "learned"}
        return cls(
            quantization_bits=int(2**max(0, min(5, v[0]))),
            group_size=int(2**max(5, min(8, v[1]))),
            fused_kernel_depth=int(max(1, min(8, v[2]))),
            scheduling_policy=sched_r.get(int(v[3]) % 3, "dynamic"),
            memory_tiling=bool(v[4] > 0.5),
            prefetch_distance=int(max(1, min(16, v[5]))),
            use_tensor_cores=bool(v[6] > 0.5),
            gradient_accumulation=int(2**max(0, min(5, v[7]))),
            activation_checkpointing=bool(v[8] > 0.5)
        )


@dataclass
class CoEvolutionGenome:
    """Hardware + Software combinés."""
    hardware: HardwareGenome = field(default_factory=HardwareGenome)
    software: SoftwareGenome = field(default_factory=SoftwareGenome)

    def to_vector(self):
        return np.concatenate([self.hardware.to_vector(), self.software.to_vector()])

    @classmethod
    def from_vector(cls, v):
        hw_len = 10  # Taille du vecteur hardware
        return cls(
            hardware=HardwareGenome.from_vector(v[:hw_len]),
            software=SoftwareGenome.from_vector(v[hw_len:])
        )


# ============================================================================
# ÉVALUATEUR CO-ÉVOLUTIF
# ============================================================================

class CoEvolutionEvaluator:
    """Évalue la paire hardware/software sur des métriques réalistes."""

    def evaluate(self, genome: CoEvolutionGenome) -> Dict:
        hw = genome.hardware
        sw = genome.software

        # 1. Performance théorique (TFLOPS)
        compute_tflops = (hw.num_compute_units * hw.unit_clock_ghz * 
                         (32 // max(1, sw.quantization_bits))) / 1000

        # 2. Bande passante effective
        bw_factor = 1.0
        if sw.memory_tiling:
            bw_factor *= 1.3
        bw_factor *= (1 + sw.prefetch_distance * 0.05)
        if sw.scheduling_policy == "dynamic":
            bw_factor *= 1.15
        effective_bw = hw.memory_bandwidth_gbs * bw_factor

        # 3. Utilisation mémoire
        base_mem_per_param = sw.quantization_bits / 8  # bytes
        if sw.group_size > 0:
            base_mem_per_param *= (1 + 4.0 / sw.group_size)  # Overhead scales
        
        total_params = 7e9  # Modèle 7B
        memory_usage_gb = total_params * base_mem_per_param / 1e9

        # 4. Latence estimée (compute-bound vs memory-bound)
        compute_time = 1000 / (compute_tflops + 1e-10)  # ms
        memory_time = memory_usage_gb / (effective_bw + 1e-10) * 1000  # ms
        
        # Roofline model
        latency_ms = max(compute_time, memory_time)

        # 5. Efficacité énergétique (estimée)
        power_watts = hw.num_compute_units * 0.5 + hw.memory_bandwidth_gbs * 0.1
        energy_per_token = power_watts * latency_ms / 1000  # Joules/token

        # 6. Pénalités et bonus
        penalty = 1.0
        if not hw.support_ternary and sw.quantization_bits <= 2:
            penalty *= 0.5  # Hardware inadapté au software
        if sw.fused_kernel_depth > hw.pipeline_depth:
            penalty *= 0.8  # Pipeline trop court pour le kernel
        if hw.interconnect_topology == "crossbar":
            penalty *= 0.9  # Crossbar cher en area

        # 7. Score fitness composite
        fitness = {
            'compute_tflops': compute_tflops,
            'effective_bw_gbs': effective_bw,
            'memory_usage_gb': memory_usage_gb,
            'latency_ms': latency_ms,
            'energy_per_token': energy_per_token,
            'penalty': penalty,
            'overall': (
                (1.0 / latency_ms) * 0.35 +  # Latence
                (1.0 / energy_per_token) * 0.25 +  # Énergie
                compute_tflops * 0.2 +  # Compute
                (1.0 / memory_usage_gb) * 0.2  # Mémoire
            ) * penalty
        }

        return fitness


# ============================================================================
# MOTEUR DE CO-ÉVOLUTION
# ============================================================================

class CoEvolutionEngine:
    def __init__(self, pop_size=80, generations=30):
        self.pop_size = pop_size
        self.generations = generations
        self.evaluator = CoEvolutionEvaluator()
        self.population = []
        self.best_genome = None
        self.best_fitness = 0.0
        self.history = []

    def initialize(self):
        print(f"\n{'='*70}")
        print(f" HARDWARE/SOFTWARE CO-EVOLUTION")
        print(f"{'='*70}\n")

        # Seeds: combinaisons connues
        seeds = [
            # GPU Standard + Software standard
            CoEvolutionGenome(
                hardware=HardwareGenome(num_compute_units=1024, unit_clock_ghz=1.5, 
                                       memory_bandwidth_gbs=1000, support_ternary=False),
                software=SoftwareGenome(quantization_bits=16, group_size=128)
            ),
            # BitNet spécialisé
            CoEvolutionGenome(
                hardware=HardwareGenome(num_compute_units=72811, unit_clock_ghz=2.0,
                                       memory_bandwidth_gbs=3000, support_ternary=True),
                software=SoftwareGenome(quantization_bits=2, group_size=128, 
                                       fused_kernel_depth=4)
            ),
            # TPU-like
            CoEvolutionGenome(
                hardware=HardwareGenome(num_compute_units=2048, unit_clock_ghz=1.2,
                                       memory_bandwidth_gbs=2000, interconnect_topology="crossbar"),
                software=SoftwareGenome(quantization_bits=8, group_size=64,
                                       scheduling_policy="static")
            ),
        ]

        self.population = seeds

        # Variantes aléatoires
        for _ in range(self.pop_size - len(seeds)):
            hw = HardwareGenome()
            sw = SoftwareGenome()
            # Randomize
            hw.num_compute_units = int(np.random.choice([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]))
            hw.unit_clock_ghz = np.random.uniform(0.5, 3.0)
            hw.support_ternary = np.random.random() < 0.5
            sw.quantization_bits = int(np.random.choice([2, 4, 8, 16]))
            sw.group_size = int(np.random.choice([32, 64, 128, 256]))
            self.population.append(CoEvolutionGenome(hw, sw))

        print(f"  Population: {len(self.population)} paires HW/SW")
        print()

    def evolve(self):
        print("Starting co-evolution...\n")
        start = time.time()

        for gen in range(self.generations):
            # Évaluer
            fitnesses = []
            for genome in self.population:
                fit = self.evaluator.evaluate(genome)
                fitnesses.append(fit)

                if fit['overall'] > self.best_fitness:
                    self.best_fitness = fit['overall']
                    self.best_genome = deepcopy(genome)
                    print(f"  ★ Gen {gen+1}: New best! Fitness={fit['overall']:.2e}")
                    print(f"    HW: {genome.hardware.num_compute_units} units, "
                          f"{genome.hardware.unit_clock_ghz:.1f}GHz, "
                          f"Ternary={genome.hardware.support_ternary}")
                    print(f"    SW: {genome.software.quantization_bits}-bit, "
                          f"Group={genome.software.group_size}, "
                          f"Fused={genome.software.fused_kernel_depth}\n")

            # Stats
            valid_fits = [f['overall'] for f in fitnesses]
            avg_fit = np.mean(valid_fits)
            self.history.append({
                'gen': gen+1,
                'best': self.best_fitness,
                'avg': avg_fit
            })

            # Sélection et reproduction (simplifié)
            selected = self._tournament_selection(fitnesses)
            next_gen = []

            # Élitisme
            elite = max(1, self.pop_size // 10)
            sorted_idx = np.argsort([f['overall'] for f in fitnesses])[::-1]
            for i in range(elite):
                next_gen.append(deepcopy(self.population[sorted_idx[i]]))

            # Reproduction
            while len(next_gen) < self.pop_size:
                p1 = self.population[np.random.choice(selected)]
                p2 = self.population[np.random.choice(selected)]
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                next_gen.append(child)

            self.population = next_gen
            print(f"  Gen {gen+1:3d}: Best={self.best_fitness:.2e}, Avg={avg_fit:.2e}, "
                  f"Time={time.time()-start:.0f}s\n")

        print(f"\n{'='*70}")
        print(f" Co-Evolution Complete ({time.time()-start:.0f}s)")
        print(f"{'='*70}\n")
        return self.best_genome, self.best_fitness

    def _tournament_selection(self, fitnesses, k=5):
        selected = []
        for _ in range(self.pop_size):
            tourney = np.random.choice(self.pop_size, k, replace=False)
            winner = max(tourney, key=lambda i: fitnesses[i]['overall'])
            selected.append(winner)
        return selected

    def _crossover(self, p1, p2):
        # Crossover séparé HW et SW
        v1 = p1.to_vector()
        v2 = p2.to_vector()
        alpha = np.random.uniform(0.3, 0.7)
        child_v = alpha * v1 + (1 - alpha) * v2
        return CoEvolutionGenome.from_vector(child_v)

    def _mutate(self, genome):
        v = genome.to_vector()
        mask = np.random.random(len(v)) < 0.2
        v[mask] *= (1 + np.random.normal(0, 0.1, len(v))[mask])
        v = np.clip(v, 0, None)
        return CoEvolutionGenome.from_vector(v)

    def print_results(self):
        if self.best_genome is None:
            return
        
        hw = self.best_genome.hardware
        sw = self.best_genome.software
        fit = self.evaluator.evaluate(self.best_genome)

        print(f"\n{'='*70}")
        print(f" OPTIMAL HARDWARE/SOFTWARE PAIR DISCOVERED")
        print(f"{'='*70}\n")
        
        print(f" HARDWARE:")
        print(f"   Compute Units:     {hw.num_compute_units:>8,}")
        print(f"   Clock:             {hw.unit_clock_ghz:>8.1f} GHz")
        print(f"   Memory BW:         {hw.memory_bandwidth_gbs:>8,} GB/s")
        print(f"   SRAM:              {hw.sram_size_kb:>8} KB")
        print(f"   Ternary Support:   {str(hw.support_ternary):>8}")
        print(f"   Topology:          {hw.interconnect_topology:>8}")
        print()
        
        print(f" SOFTWARE:")
        print(f"   Quantization:      {sw.quantization_bits:>8}-bit")
        print(f"   Group Size:        {sw.group_size:>8}")
        print(f"   Kernel Fusion:     {sw.fused_kernel_depth:>8} ops")
        print(f"   Scheduling:        {sw.scheduling_policy:>8}")
        print(f"   Prefetch:          {sw.prefetch_distance:>8}")
        print()
        
        print(f" PERFORMANCE:")
        print(f"   Compute:           {fit['compute_tflops']:>8.1f} TFLOPS")
        print(f"   Bandwidth:         {fit['effective_bw_gbs']:>8.0f} GB/s")
        print(f"   Memory Usage:      {fit['memory_usage_gb']:>8.1f} GB")
        print(f"   Latency:           {fit['latency_ms']:>8.2f} ms")
        print(f"   Energy/Token:      {fit['energy_per_token']:>8.3f} J")
        print(f"   Overall Fitness:   {fit['overall']:>8.2e}")
        print()


if __name__ == "__main__":
    engine = CoEvolutionEngine(pop_size=60, generations=25)
    engine.initialize()
    best, fit = engine.evolve()
    engine.print_results()
