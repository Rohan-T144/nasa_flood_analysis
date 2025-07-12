# scene5_spiking_self_attention_v0_19_resized.py
from manim import *
import numpy as np

class SpikingSelfAttentionScene(Scene):
    def construct(self):
        # --- Overall Scaling and Layout ---
        scale = 0.6 # Master scale factor for all elements
        h_buff = 0.7 # Horizontal buffer between elements
        
        title = Text("Spiking Self-Attention Module").to_edge(UP)
        self.play(Write(title))

        # --- Define All Components First ---
        # This approach makes layout management much easier.

        # (a) Input and Q, K, V Tensors
        input_tensor = Rectangle(width=1.5*scale, height=2.5*scale, color=BLUE, fill_opacity=0.3)
        input_label = Text("Input", font_size=20*scale).next_to(input_tensor, DOWN, buff=0.2*scale)
        input_group = VGroup(input_tensor, input_label)

        q_tensor = input_tensor.copy().set_color(GREEN)
        k_tensor = input_tensor.copy().set_color(YELLOW)
        v_tensor = input_tensor.copy().set_color(ORANGE)

        q_label = Text("Query (Q)", font_size=20*scale).next_to(q_tensor, DOWN, buff=0.2*scale)
        k_label = Text("Key (K)", font_size=20*scale).next_to(k_tensor, DOWN, buff=0.2*scale)
        v_label = Text("Value (V)", font_size=20*scale).next_to(v_tensor, DOWN, buff=0.2*scale)

        q_group = VGroup(q_tensor, q_label)
        k_group = VGroup(k_tensor, k_label)
        v_group = VGroup(v_tensor, v_label)
        qkv_tensors = VGroup(q_group, k_group, v_group).arrange(DOWN, buff=0.4*scale)

        # (b) Attention Scores
        attn_scores = Square(side_length=2*scale, color=PURPLE, fill_opacity=0.4)
        attn_label = Text("Attn Scores", font_size=20*scale).next_to(attn_scores, DOWN, buff=0.2*scale)
        attn_group = VGroup(attn_scores, attn_label)

        # (c) Spiking Activation & Sparse Attention
        attn_lif = Square(side_length=2*scale, color=BLUE, fill_opacity=0.2)
        lif_label = Text("attn_lif", font_size=20*scale).next_to(attn_lif, DOWN, buff=0.2*scale)
        lif_group = VGroup(attn_lif, lif_label)

        sparse_attn = attn_lif.copy().set_color(WHITE)
        sparse_label = Text("Spiking Attn", font_size=20*scale).next_to(sparse_attn, DOWN, buff=0.2*scale)
        sparse_group = VGroup(sparse_attn, sparse_label)

        # (d) Attended Features
        attended_features = v_tensor.copy().set_color(TEAL)
        attended_label = Text("Attended Features", font_size=20*scale).next_to(attended_features, DOWN, buff=0.2*scale)
        attended_group = VGroup(attended_features, attended_label)
        
        # --- Arrange Layout in a Master VGroup ---
        diagram = VGroup(
            input_group,
            qkv_tensors,
            attn_group,
            lif_group,
            sparse_group,
            attended_group
        ).arrange(RIGHT, buff=h_buff).center()

        # --- Define Connecting Mobjects (Arrows, Text) ---
        q_path = Arrow(input_group.get_right(), q_group.get_left(), buff=0.1, stroke_width=3*scale, max_tip_length_to_length_ratio=0.15)
        k_path = Arrow(input_group.get_right(), k_group.get_left(), buff=0.1, stroke_width=3*scale, max_tip_length_to_length_ratio=0.15)
        v_path = Arrow(input_group.get_right(), v_group.get_left(), buff=0.1, stroke_width=3*scale, max_tip_length_to_length_ratio=0.15)
        conv_text = Text("1x1 Conv", font_size=16*scale).next_to(k_path, UP, buff=0.05)
        
        matmul_op = MathTex(r"Q \cdot K^T", font_size=30*scale).next_to(attn_group, LEFT*0.3, buff=h_buff/3)
        arrow_to_lif = Arrow(attn_group.get_right(), lif_group.get_left(), buff=0.1, stroke_width=3*scale, max_tip_length_to_length_ratio=0.15)
        arrow_to_sparse = Arrow(lif_group.get_right(), sparse_group.get_left(), buff=0.1, stroke_width=3*scale, max_tip_length_to_length_ratio=0.15)
        matmul_op2 = MathTex(r"\times V", font_size=30*scale).next_to(sparse_group, RIGHT, buff=h_buff/3)

        # --- ANIMATION SEQUENCE ---
        
        # (a) Q, K, V Generation
        # Narration: "Our model includes a custom Spiking Self-Attention block. It starts with an input spike tensor."
        self.play(Write(input_group))
        # Narration: "This input is passed through three separate 1x1 convolutions to generate three distinct tensors: Query, Key, and Value."
        self.play(
            LaggedStart(
                FadeIn(qkv_tensors),
                GrowArrow(q_path), GrowArrow(k_path), GrowArrow(v_path),
                Write(conv_text),
                lag_ratio=0.5
            )
        )
        self.wait(1)

        # (b) Attention Scores
        # Narration: "Next, a matrix multiplication is performed between the Query and Key tensors to calculate raw attention scores."
        self.play(FadeIn(attn_group, shift=RIGHT), Write(matmul_op))
        self.wait(1)

        # (c) The Spiking Activation
        # Narration: "This is the critical step. These scores are not used directly. Instead, they become the input voltage to a grid of LIF neurons."
        self.play(FadeIn(lif_group, shift=RIGHT), GrowArrow(arrow_to_lif))
        self.wait(0.5)

        # Narration: "Over T timesteps, these neurons integrate the scores. Only neurons with high enough integrated potential will fire."
        # Narration: "This produces a very sparse matrix of spikes, our 'Spiking Attention' map."
        spikes = VGroup(*[Dot(radius=0.05*scale, color=YELLOW).move_to(
            lif_group.get_center() + np.random.uniform(-0.5, 0.5, 3)*scale
        ) for _ in range(7)])
        self.play(LaggedStart(*[Flash(s, line_length=0.2, time_width=0.5, color=YELLOW) for s in spikes], lag_ratio=0.1))
        
        self.play(
            FadeIn(sparse_group, shift=RIGHT),
            GrowArrow(arrow_to_sparse),
            Transform(spikes, sparse_group[0])
        )
        self.wait(1)
        
        # (d) Applying Attention
        # Narration: "This sparse spike map is then multiplied with the Value tensor."
        self.play(FadeOut(spikes))
        v_copy = v_group[0].copy()
        sparse_copy = sparse_group[0].copy()
        # Narration: "This efficiently selects only the values corresponding to where the attention spikes fired."
        self.play(
            FadeIn(attended_group, shift=RIGHT),
            Write(matmul_op2),
            Transform(VGroup(v_copy, sparse_copy), attended_group[0])
        )
        self.wait(1)

        # (e) Output with Residual Connection
        final_group = VGroup(input_group, attended_group)
        residual_path = ArcBetweenPoints(input_group.get_bottom(), attended_group.get_bottom(), angle=-PI/2, color=GREY)
        plus_sign = MathTex("+", font_size=30*scale).next_to(residual_path, DOWN, buff=0.1)
        
        # Narration: "Finally, the output is added back to the original input via a residual connection, completing the block."
        self.play(
            FadeOut(qkv_tensors, attn_group, lif_group, sparse_group, q_path, k_path, v_path, conv_text, matmul_op, matmul_op2, arrow_to_lif, arrow_to_sparse, v_copy, sparse_copy),
            final_group.animate.arrange(RIGHT, buff=2.5)
        )
        self.play(Create(residual_path), Write(plus_sign))
        self.wait(2)
