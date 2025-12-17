# E. Tampilkan Hasil dengan UI Menarik
        st.divider()
        st.subheader("📊 Hasil Analisis Kesehatan Tidur")
        
        # Buat kolom untuk Ringkasan
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            if result_label.lower() in ["healthy", "none", "normal"]:
                st.success(f"### Kondisi Anda: **{result_label}** ✨")
                st.write("Tetap pertahankan pola hidup sehatmu! Tidur yang cukup adalah investasi terbaik untuk hari esok.")
                st.balloons()
            elif result_label == "Insomnia":
                st.warning(f"### Kondisi Anda: **{result_label}** ⚠️")
                st.write("Sepertinya Anda memiliki gejala sulit tidur. Jangan terlalu cemas, banyak orang mengalaminya dan ini bisa diperbaiki.")
            elif result_label == "Sleep Apnea":
                st.error(f"### Kondisi Anda: **{result_label}** 🚨")
                st.write("Kondisi ini berkaitan dengan gangguan pernapasan saat tidur. Sangat disarankan untuk memantau kualitas tidur lebih serius.")

        with res_col2:
            st.metric(label="Tingkat Keyakinan AI", value=f"{confidence:.1f}%")

        # F. SEKSI SARAN & KATA PENYEMANGAT
        st.write("---")
        st.write("### 💡 Saran Untuk Anda:")
        
        if result_label == "Insomnia":
            st.markdown("""
            * **Rutinitas Tidur:** Cobalah tidur dan bangun di jam yang sama setiap hari.
            * **Hindari Gadget:** Matikan layar 30-60 menit sebelum tidur karena cahaya biru bisa menghambat melatonin.
            * **Relaksasi:** Coba teknik pernapasan atau mendengarkan musik tenang sebelum memejamkan mata.
            * *Semangat! Insomnia bukan akhir dari segalanya, tubuhmu hanya butuh waktu untuk belajar rileks kembali.*
            """)
        elif result_label == "Sleep Apnea":
            st.markdown("""
            * **Posisi Tidur:** Cobalah tidur menyamping untuk membantu menjaga saluran udara tetap terbuka.
            * **Kontrol Berat Badan:** Karena data BMI Anda berpengaruh, menjaga berat badan ideal bisa sangat membantu.
            * **Konsultasi Dokter:** Kami sangat menyarankan untuk melakukan *Sleep Study* atau berkonsultasi dengan spesialis THT/Paru.
            * *Jangan menyerah! Dengan penanganan yang tepat, Anda akan kembali bangun dengan perasaan segar setiap pagi.*
            """)
        else: # Healthy
            st.markdown("""
            * **Manajemen Stres:** Terus kelola tingkat stres Anda karena ini kunci tidur berkualitas.
            * **Aktivitas Fisik:** Olahraga ringan di pagi atau sore hari akan membantu tidur malam lebih nyenyak.
            * *Kesehatan adalah kekayaan paling berharga. Teruslah menginspirasi orang sekitar dengan energi positifmu!*
            """)

        st.info("⚠️ **Catatan:** Hasil ini adalah prediksi AI berdasarkan data yang Anda masukkan. Silakan hubungi tenaga medis profesional untuk diagnosis resmi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
