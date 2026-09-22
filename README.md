# Custom Generative AI Studio

Studio AI kustom berbasis lokal untuk Virtual Try-On (kamar pas baju AI), Generative Inpainting (tambah objek, ubah suasana, atau tempatkan karakter waifu), serta Ganti Background instan. Dioptimalkan untuk GPU laptop (NVIDIA GeForce RTX 4050 6GB VRAM).

## Fitur Utama

1. **Virtual Try-On (Ganti Baju):** Mengganti pakaian pada foto tubuh menggunakan foto katalog baju referensi dengan preservasi wajah dan pose.
2. **Generatif Inpainting & Objek / Waifu / Suasana:** Coret area pada kanvas, masukkan prompt dan foto referensi (IP-Adapter) untuk menempatkan karakter atau mengubah suasana ruangan.
3. **Ganti Background Instan:** Segmentasi otomatis menggunakan Rembg untuk memotong latar belakang dan menyatukannya dengan pemandangan baru dalam hitungan detik.
4. **Dukungan Akses Jaringan Lokal:** Dapat diakses langsung dari smartphone (Android/iOS) pada jaringan Wi-Fi yang sama melalui browser.

## Persyaratan Sistem

* Windows 10/11
* Python 3.10 atau 3.11
* GPU NVIDIA dengan VRAM minimal 6 GB
* CUDA 12.1+

## Cara Menjalankan

1. Jalankan `run.bat` (atau aktifkan virtual environment lalu jalankan `python app.py`).
2. Buka browser:
   * Laptop: `http://localhost:7860`
   * HP (Wi-Fi sama): `http://<IP_LAPTOP>:7860`
