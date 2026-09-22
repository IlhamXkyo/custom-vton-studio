import os
import socket
import gradio as gr
from PIL import Image
from pipeline_manager import (
    virtual_try_on,
    inpaint_general,
    replace_background
)

def get_local_ip():
    """Retrieve local Wi-Fi / LAN IP address for Android connectivity."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# UI Theme & CSS
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.tab-header {
    font-size: 1.15rem;
    font-weight: 600;
}
.banner-info {
    padding: 12px;
    background: #2b313e;
    color: #f1f5f9;
    border-radius: 8px;
    margin-bottom: 12px;
}
"""

with gr.Blocks(title="Custom Generative AI Studio") as demo:
    local_ip = get_local_ip()
    
    gr.Markdown(f"""
    # 🎨 Custom Generative AI Studio
    **Studio Pemroses Kreatif Lokal (RTX 4050 6GB VRAM Safe)**  
    *Akses dari browser laptop: `http://localhost:7860` | Akses dari HP Android (Wi-Fi sama): `http://{local_ip}:7860`*
    """)

    with gr.Tabs():
        # TAB 1: Virtual Try-On
        with gr.TabItem("👗 Virtual Try-On (Ganti Baju)", id="tab_vton"):
            gr.Markdown("### Ganti baju pada badanmu menggunakan foto pakaian referensi")
            with gr.Row():
                with gr.Column(scale=5):
                    vton_person = gr.ImageEditor(
                        label="Foto Kamu (Coret area baju yang ingin diganti)",
                        type="pil",
                        brush=gr.Brush(colors=["#ffffff"], default_size=25),
                        sources=["upload", "webcam"]
                    )
                with gr.Column(scale=4):
                    vton_garment = gr.Image(
                        label="Foto Pakaian Referensi (Dari Olshop / Katalog)",
                        type="pil",
                        sources=["upload"]
                    )
                    vton_prompt = gr.Textbox(
                        label="Keterangan Pakaian Tambahan (Opsional)",
                        value="wearing this clothing, perfectly fitting, realistic cloth texture, natural lighting and folds",
                        lines=2
                    )
                    with gr.Accordion("Pengaturan Lanjutan", open=False):
                        vton_strength = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Kekuatan Kemiripan Baju (IP-Adapter)")
                        vton_steps = gr.Slider(15, 40, value=25, step=1, label="Langkah Inferensi (Steps)")
                    vton_btn = gr.Button("✨ Coba Baju Sekarang", variant="primary", size="lg")
                
                with gr.Column(scale=5):
                    vton_output = gr.Image(label="Hasil Foto Coba Baju", type="pil")

            vton_btn.click(
                fn=virtual_try_on,
                inputs=[vton_person, vton_garment, vton_prompt, gr.State("bad quality, blurry, deformed"), vton_strength, vton_steps],
                outputs=[vton_output]
            )

        # TAB 2: Generative Inpainting & Objek / Waifu / Suasana
        with gr.TabItem("🪄 Generatif Studio (Objek / Suasana / Waifu)", id="tab_inpaint"):
            gr.Markdown("### Ganti objek, ubah suasana ruangan, atau tempatkan karakter waifu di area coretan")
            with gr.Row():
                with gr.Column(scale=5):
                    inpaint_canvas = gr.ImageEditor(
                        label="Foto Dasar (Coret area yang ingin ditambah / diganti)",
                        type="pil",
                        brush=gr.Brush(colors=["#ffffff"], default_size=30),
                        sources=["upload", "webcam"]
                    )
                with gr.Column(scale=4):
                    inpaint_ref = gr.Image(
                        label="Foto Referensi (Opsional: Masukkan foto Waifu / Produk agar mirip)",
                        type="pil",
                        sources=["upload"]
                    )
                    inpaint_prompt = gr.Textbox(
                        label="Deskripsi Objek / Suasana / Pose",
                        placeholder="Contoh: cute anime girl waifu sitting next to me, smiling, highly detailed ATAU cyberpunk city atmosphere, neon reflections",
                        lines=3
                    )
                    inpaint_neg = gr.Textbox(
                        label="Negative Prompt",
                        value="bad quality, blurry, deformed, disfigured, distorted, lowres",
                        lines=1
                    )
                    with gr.Accordion("Pengaturan Lanjutan", open=False):
                        inpaint_ref_scale = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Kekuatan Referensi Foto (IP-Adapter)")
                        inpaint_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Kepatuhan Prompt (Guidance Scale)")
                        inpaint_steps = gr.Slider(15, 50, value=25, step=1, label="Langkah Inferensi (Steps)")
                        inpaint_seed = gr.Number(value=-1, label="Seed (-1 untuk acak)")
                    inpaint_btn = gr.Button("🔮 Generate Objek / Karakter", variant="primary", size="lg")

                with gr.Column(scale=5):
                    inpaint_output = gr.Image(label="Hasil Generatif", type="pil")

            inpaint_btn.click(
                fn=inpaint_general,
                inputs=[
                    inpaint_canvas,
                    inpaint_prompt,
                    inpaint_neg,
                    inpaint_ref,
                    inpaint_ref_scale,
                    inpaint_cfg,
                    inpaint_steps,
                    inpaint_seed
                ],
                outputs=[inpaint_output]
            )

        # TAB 3: Instant Background Replacement
        with gr.TabItem("🖼️ Ganti Background Instan", id="tab_bg"):
            gr.Markdown("### Ganti latar belakang foto kamu secara instan dengan foto pemandangan kustom")
            with gr.Row():
                with gr.Column(scale=4):
                    bg_person = gr.Image(label="Foto Diri", type="pil", sources=["upload", "webcam"])
                with gr.Column(scale=4):
                    bg_new = gr.Image(label="Foto Latar Belakang Baru", type="pil", sources=["upload"])
                    bg_edge = gr.Slider(0, 10, value=3, step=1, label="Kehalusan Garis Tepi (Edge Smoothing)")
                    bg_btn = gr.Button("⚡ Ganti Latar Belakang", variant="primary", size="lg")
                with gr.Column(scale=5):
                    bg_output = gr.Image(label="Hasil Komposisi", type="pil")

            bg_btn.click(
                fn=replace_background,
                inputs=[bg_person, bg_new, bg_edge],
                outputs=[bg_output]
            )

if __name__ == "__main__":
    print(f"Starting server on port 7860...")
    print(f"Local URL: http://localhost:7860")
    print(f"Network URL (for Android / devices on same Wi-Fi): http://{get_local_ip()}:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)
