## 🚀 Cara Deploy & Integrasi Backend FastAPI di Railway

1. **Deploy Backend ke Railway**
   - https://github.com/devilk1d/telvora-model-service (Push kode backend ke GitHub) 
   - Buat project baru di Railway, hubungkan ke repository backend Anda.
   - Railway akan otomatis build dan deploy backend FastAPI.

2. **Generate Public URL**
   - Setelah deploy, klik tombol "Generate Domain" di dashboard Railway.
   - Railway akan memberikan public URL (misal: https://your-service.up.railway.app).

3. **Integrasi dengan Frontend**
   - Copy public URL Railway.
   - Masukkan ke environment variable frontend (misal: VITE_RECSYS_URL).
   - Frontend dapat mengakses API backend melalui URL tersebut.

4. **Monitoring & Logs**
   - Railway menyediakan dashboard untuk melihat status service, logs, dan error.

Ringkasnya: Railway memudahkan deploy backend, generate public URL, dan integrasi dengan frontend tanpa perlu setup server manual.