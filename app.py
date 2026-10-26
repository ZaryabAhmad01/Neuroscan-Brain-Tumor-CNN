
        for index in np.argsort(probabilities)[::-1]:
            name = CLASS_NAMES[int(index)]
            score = float(probabilities[int(index)])

            st.markdown(
                f"**{DISPLAY_NAMES[name]}**"
                + (" · ★ Top score" if name == predicted_class else "")
                + f" &nbsp; **{score:.2f}%**"
            )

            # st.progress takes a value from 0.0 to 1.0.
            # This avoids custom HTML span rendering issues.
            st.progress(min(max(score / 100.0, 0.0), 1.0))


    with prob_right:
        st.markdown(
            '<div class="panel-label">PROBABILITY COMPARISON</div>',
            unsafe_allow_html=True,
        )

        chart_data = pd.DataFrame({
            "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Probability (%)": [
                float(probabilities[i]) for i in range(len(CLASS_NAMES))
            ],
        }).set_index("Class")

        # Compatible built-in chart. Do not pass color="#hex" here:
        # some Streamlit versions interpret it as a column encoding.
        st.bar_chart(
            chart_data,
            height=280,
        )



    # ========================================================
    # 9. DOWNLOADABLE SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">04 / Prediction summary</div>',
        unsafe_allow_html=True,
    )

    summary = {
        "Image filename": uploaded_file.name,
        "Predicted class": DISPLAY_NAMES[predicted_class],
        "Top-class score (%)": round(confidence, 2),
        "Inference time (seconds)": round(inference_time, 4),
    }

    for i, name in enumerate(CLASS_NAMES):
        summary[f"{DISPLAY_NAMES[name]} score (%)"] = round(
            float(probabilities[i]), 4
        )

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Field", "Value"],
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download prediction summary (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="neuroscan_prediction_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# 10. DISCLAIMER AND FOOTER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <b>Important medical disclaimer</b><br>
        NeuroScan AI is an educational and research demonstration,
        not a validated diagnostic system. Model scores may be wrong.
        Do not use this application to make clinical decisions.
        MRI scans should be interpreted by qualified healthcare
        professionals.
    </div>

    <div class="footer">
        <b>NeuroScan AI</b> · Brain MRI Classification Dashboard<br>
        Powered by Streamlit, TensorFlow, Keras and a custom CNN.<br>
        For educational and research purposes only.
    </div>
    """,
    unsafe_allow_html=True,
)

