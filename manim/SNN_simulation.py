from manim import *
import numpy as np

class Neuron(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.circle = Circle(radius=0.3, color=BLUE_E)
        self.circle.set_fill(BLUE_E, opacity=1)
        self.circle.set_stroke(BLUE, opacity=0.8)

        # --- LIF Model Parameters ---
        self.potential = ValueTracker(0)
        self.threshold = 1.0
        self.beta = 0.95  # Decay factor
        self.is_spiking = False
        
        self.add(self.circle)

    def update_potential(self, input_current):
        """Updates potential based on input and decay. Sets spiking flag."""
        if self.is_spiking:
            # Reset after a spike
            self.potential.set_value(0)
            self.is_spiking = False
        else:
            # Update based on decay and input
            new_potential = self.potential.get_value() * self.beta + input_current
            self.potential.set_value(new_potential)
            
            # Check for spike
            if self.potential.get_value() >= self.threshold:
                self.is_spiking = True
    
    def get_color_for_potential(self):
        """Returns the color corresponding to the current potential."""
        potential_ratio = np.clip(self.potential.get_value() / self.threshold, 0, 1)
        return interpolate_color(BLUE_E, YELLOW, potential_ratio)

class Synapse(VMobject):
    def __init__(self, pre_neuron: Neuron, post_neuron: Neuron, **kwargs):
        super().__init__(**kwargs)
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = np.random.uniform(0.1, 0.5)
        self.color = BLUE # Resting color

        self.line = Line(
            self.pre.circle.get_center() + self._offset(self.pre, self.post),
            self.post.circle.get_center() - self._offset(self.pre, self.post),
            stroke_color=self.color,
            stroke_opacity=0.8,
            stroke_width=1 + 6 * self.weight
        )
        self.add(self.line)

    def _offset(self, source, target):
        direction = target.circle.get_center() - source.circle.get_center()
        return (direction / np.linalg.norm(direction)) * 0.3

class SpikingNeuralNetwork(Scene):
    def construct(self):
        # --- Network Setup ---
        input_layer = [Neuron() for _ in range(3)]
        hidden_layer = [Neuron() for _ in range(4)]
        output_layer = [Neuron() for _ in range(2)]
        
        # Position neurons
        for i, neuron in enumerate(input_layer):
            neuron.move_to(LEFT * 5 + DOWN * (i - 1) * 1.5)
        for i, neuron in enumerate(hidden_layer):
            neuron.move_to(RIGHT * 0 + DOWN * (i - 1.5) * 1.2)
        for i, neuron in enumerate(output_layer):
            neuron.move_to(RIGHT * 5 + DOWN * (i - 0.5) * 1.5)

        # Create synapses
        synapses_h1 = [Synapse(pre, post) for pre in input_layer for post in hidden_layer]
        synapses_h2 = [Synapse(pre, post) for pre in hidden_layer for post in output_layer]
        all_synapses = synapses_h1 + synapses_h2
        
        # Initial display
        self.play(
            *[Create(s.line) for s in all_synapses],
            *[Create(n) for n in (input_layer + hidden_layer + output_layer)],
            run_time=1.5
        )
        
        time_step_counter = ValueTracker(0)
        time_display = VGroup(
            Text("Time Step: ", font_size=24),
            DecimalNumber(0, num_decimal_places=0, font_size=24)
            .add_updater(lambda v: v.set_value(time_step_counter.get_value()))
        ).arrange(RIGHT).to_corner(UR)
        self.add(time_display)
        self.wait(1)

        # --- Simulation Loop ---
        simulation_steps = 25
        for time_step in range(simulation_steps):
            time_step_counter.set_value(time_step + 1)
            
            # --- Step 1: Update Input Layer and Animate Color ---
            input_anims = []
            for neuron in input_layer:
                # Calculate new potential but don't animate yet
                neuron.update_potential(max(0, np.random.normal(0.1, 0.15)))
                # Create an animation for the color change
                input_anims.append(neuron.circle.animate.set_fill(neuron.get_color_for_potential()))
            
            # Play the color change animation for the input layer
            if input_anims:
                self.play(*input_anims, run_time=0.4)

            spiking_input_neurons = [n for n in input_layer if n.is_spiking]
            
            # --- Step 2: Propagate Input -> Hidden ---
            self.propagate_and_animate(spiking_input_neurons, synapses_h1, hidden_layer)
            
            spiking_hidden_neurons = [n for n in hidden_layer if n.is_spiking]
            
            # --- Step 3: Propagate Hidden -> Output ---
            self.propagate_and_animate(spiking_hidden_neurons, synapses_h2, output_layer)

            # --- Step 4: Animate Final Output Spikes ---
            spiking_output_neurons = [n for n in output_layer if n.is_spiking]
            if spiking_output_neurons:
                final_anims = []
                for n in spiking_output_neurons:
                    final_anims.append(Flash(n.circle, color=RED, flash_radius=0.6))
                    n.update_potential(0) # Reset state for next loop
                self.play(AnimationGroup(*final_anims, lag_ratio=0.2), run_time=0.8)

            # --- Step 5: Universal Decay Step ---
            decay_anims = []
            all_postsynaptic_neurons = hidden_layer + output_layer
            neurons_with_input = {s.post for s in all_synapses if s.pre.is_spiking}

            for neuron in all_postsynaptic_neurons:
                if neuron not in neurons_with_input and not neuron.is_spiking:
                    neuron.update_potential(0) # Apply decay
                    decay_anims.append(neuron.circle.animate.set_fill(neuron.get_color_for_potential()))
            
            if decay_anims:
                self.play(*decay_anims, run_time=0.4)
            else:
                self.wait(0.2)

        self.wait(2)

    def propagate_and_animate(self, spiking_neurons, synapses, next_layer):
        if not spiking_neurons:
            return

        # Part 1: The spike event (flash and synapse activation)
        spike_anims = []
        firing_synapses = [s for s in synapses if s.pre in spiking_neurons]
        for n in spiking_neurons:
            spike_anims.append(Flash(n.circle, color=YELLOW, flash_radius=0.5))
        for s in firing_synapses:
            spike_anims.append(s.line.animate.set_color(YELLOW))
        self.play(*spike_anims, run_time=0.4)

        # Part 2: Propagation (revert colors and update next layer's potential)
        prop_anims = []
        
        # Reset presynaptic neurons that just fired
        for n in spiking_neurons:
            n.update_potential(0) # This resets potential to 0 and is_spiking to False
            prop_anims.append(n.circle.animate.set_fill(BLUE_E))

        # Reset synapses to their resting color
        for s in firing_synapses:
            prop_anims.append(s.line.animate.set_color(s.color))

        # Update and animate postsynaptic neurons that receive input
        incoming_currents = {n: 0 for n in next_layer}
        for s in firing_synapses:
            incoming_currents[s.post] += s.weight

        for n, current in incoming_currents.items():
            if current > 0:
                n.update_potential(current) # Update potential and possibly set is_spiking
                prop_anims.append(n.circle.animate.set_fill(n.get_color_for_potential()))
        
        if prop_anims:
            self.play(*prop_anims, run_time=0.6)
