def get_html_interpretations(wc, mu_p, horizon, risk_profile, lang="ES"):
    """Devuelve interpretaciones en HTML con formato <strong>...</strong> para PDF o web."""

    # Interpretación del VaR
    if lang == "ES":
        var_text = (
            f"El <strong>VaR al 95 %</strong> indica que, en un escenario adverso, podrías perder hasta "
            f"<strong>{abs(wc):,.0f} €</strong> en los próximos <strong>{horizon} días hábiles</strong>. "
            f"Este valor refleja el riesgo potencial de tu cartera con perfil <strong>{risk_profile}</strong>."
        )
    else:
        var_text = (
            f"The <strong>95% VaR</strong> suggests that in a worst-case scenario, you could lose up to "
            f"<strong>€{abs(wc):,.0f}</strong> over the next <strong>{horizon} trading days</strong>. "
            f"This reflects the downside risk of your <strong>{risk_profile}</strong> profile."
        )

    # Interpretación de la ganancia esperada
    if lang == "ES":
        mu_text = (
            f"La <strong>ganancia esperada</strong> para este mismo periodo es de aproximadamente "
            f"<strong>{mu_p:,.0f} €</strong>. Esto representa el escenario medio estimado para tu cartera "
            f"en base a su comportamiento histórico reciente."
        )
    else:
        mu_text = (
            f"The <strong>expected average gain</strong> for this horizon is approximately "
            f"<strong>€{mu_p:,.0f}</strong>. This represents the estimated mean scenario for your portfolio "
            f"based on recent historical performance."
        )

    return var_text, mu_text
