"""
Theme helpers for MedPredict.
Provides colour palette and small reusable styled-component functions.
All HTML uses <div> instead of <hr> to avoid Streamlit rendering bugs.
"""

PALETTE = {
    "primary":        "#2B6CB0",
    "primary_dark":   "#2C5282",
    "primary_light":  "#4299E1",
    "secondary":      "#38A169",
    "success":        "#38A169",
    "warning":        "#D69E2E",
    "danger":         "#E53E3E",
    "text_primary":   "#1A202C",
    "text_secondary": "#4A5568",
    "text_muted":     "#718096",
    "text_light":     "#A0AEC0",
    "bg_card":        "#FFFFFF",
    "bg_card_alt":    "#F7FAFC",
    "bg_page":        "#F0F4F8",
    "border":         "#E2E8F0",
    "chart_sequence": [
        "#2B6CB0", "#38A169", "#D69E2E", "#E53E3E", "#805AD5",
        "#DD6B20", "#319795", "#3182CE", "#48BB78", "#ECC94B",
    ],
}


def get_plotly_template():
    return "plotly_white"


# ── Styled Components ────────────────────────────────────────────

def styled_metric_card(label, value, color=None):
    c = color or PALETTE["primary"]
    return (
        f'<div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;'
        f'padding:1.1rem 1.2rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);">'
        f'<p style="margin:0;font-size:0.8rem;color:#718096;">{label}</p>'
        f'<p style="margin:0.25rem 0 0;font-size:1.4rem;font-weight:700;color:{c};">{value}</p>'
        f'</div>'
    )


def styled_info_box(title, body):
    return (
        f'<div style="background:#EBF8FF;border-left:4px solid #2B6CB0;'
        f'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:0.75rem 0;">'
        f'<p style="margin:0 0 0.3rem;font-weight:600;color:#2C5282;font-size:0.88rem;">{title}</p>'
        f'<p style="margin:0;color:#2D3748;font-size:0.85rem;line-height:1.55;">{body}</p>'
        f'</div>'
    )


def styled_warning_box(body):
    return (
        f'<div style="background:#FFFAF0;border-left:4px solid #D69E2E;'
        f'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin:0.75rem 0;">'
        f'<p style="margin:0;color:#744210;font-size:0.84rem;line-height:1.55;">{body}</p>'
        f'</div>'
    )
