#include "p_roboai_viz/displays_panel.hpp"

#include <QColorDialog>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QVBoxLayout>

namespace p_roboai_viz {

// ── DisplayRow ────────────────────────────────────────────────────────────────

DisplayRow::DisplayRow(const QString& id,
                       const QString& label,
                       const QString& topic,
                       QColor         color,
                       QWidget*       parent)
    : QWidget(parent), _id(id), _color(color)
{
    setFixedHeight(58);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(4, 4, 4, 4);
    root->setSpacing(2);

    // ── top row: checkbox + colour swatch + label ──────────────────────────
    auto* top = new QHBoxLayout;
    top->setSpacing(6);

    _check = new QCheckBox(label, this);
    _check->setChecked(true);
    _check->setStyleSheet("font-weight: bold; font-size: 12px;");

    _color_btn = new QLabel(this);
    _color_btn->setFixedSize(18, 18);
    _color_btn->setCursor(Qt::PointingHandCursor);
    _color_btn->setToolTip("Click to change colour");
    QString swatch = QString("background:%1; border:1px solid #888; border-radius:3px;")
                         .arg(color.name());
    _color_btn->setStyleSheet(swatch);
    _color_btn->installEventFilter(this);

    auto* topic_lbl = new QLabel(topic, this);
    topic_lbl->setStyleSheet("color:#888; font-size:10px;");
    topic_lbl->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    top->addWidget(_check);
    top->addStretch();
    top->addWidget(topic_lbl);
    top->addWidget(_color_btn);

    // ── bottom row: opacity slider ─────────────────────────────────────────
    auto* bot = new QHBoxLayout;
    bot->setSpacing(6);

    auto* op_lbl = new QLabel("Opacity:", this);
    op_lbl->setStyleSheet("color:#aaa; font-size:10px;");

    _opacity = new QSlider(Qt::Horizontal, this);
    _opacity->setRange(10, 100);
    _opacity->setValue(90);
    _opacity->setFixedHeight(16);
    _opacity->setStyleSheet(
        "QSlider::groove:horizontal { height:4px; background:#555; border-radius:2px; }"
        "QSlider::handle:horizontal  { width:12px; height:12px; margin:-4px 0;"
        "  background:#4a9; border-radius:6px; }"
        "QSlider::sub-page:horizontal { background:#4a9; border-radius:2px; }");

    bot->addWidget(op_lbl);
    bot->addWidget(_opacity, 1);

    root->addLayout(top);
    root->addLayout(bot);

    // ── connections ────────────────────────────────────────────────────────
    connect(_check,   &QCheckBox::toggled, this, [this](bool){ emit layerChanged(_id); });
    connect(_opacity, &QSlider::valueChanged, this, [this](int){ emit layerChanged(_id); });
}

bool DisplayRow::isEnabled() const { return _check->isChecked(); }

float DisplayRow::opacity() const { return _opacity->value() / 100.0f; }

bool DisplayRow::eventFilter(QObject* obj, QEvent* ev)
{
    if (obj == _color_btn && ev->type() == QEvent::MouseButtonPress) {
        pickColor();
        return true;
    }
    return QWidget::eventFilter(obj, ev);
}

void DisplayRow::pickColor()
{
    QColor c = QColorDialog::getColor(_color, this, "Choose layer colour");
    if (c.isValid()) {
        _color = c;
        _color_btn->setStyleSheet(
            QString("background:%1; border:1px solid #888; border-radius:3px;")
                .arg(c.name()));
        emit layerChanged(_id);
    }
}

// ── DisplaysPanel ─────────────────────────────────────────────────────────────

DisplaysPanel::DisplaysPanel(QWidget* parent) : QWidget(parent)
{
    setFixedWidth(230);
    setStyleSheet("background:#2a2a2a;");

    auto* outer = new QVBoxLayout(this);
    outer->setContentsMargins(0, 0, 0, 0);
    outer->setSpacing(0);

    // ── title bar ──────────────────────────────────────────────────────────
    auto* title = new QLabel("  Displays", this);
    title->setFixedHeight(32);
    title->setStyleSheet(
        "background:#1e1e1e; color:#ddd; font-size:13px; font-weight:bold;"
        "border-bottom:1px solid #444;");

    // ── scroll area with rows ──────────────────────────────────────────────
    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setStyleSheet("QScrollArea { border:none; background:#2a2a2a; }");

    auto* container = new QWidget;
    container->setStyleSheet("background:#2a2a2a;");
    auto* rows_layout = new QVBoxLayout(container);
    rows_layout->setContentsMargins(6, 6, 6, 6);
    rows_layout->setSpacing(2);

    // ── define the display layers ──────────────────────────────────────────
    struct LayerDef {
        const char* id;
        const char* label;
        const char* topic;
        QColor      color;
    };
    static const LayerDef defs[] = {
        { "map",       "Map",        "/p_roboai_slam/map",         QColor(200,200,200) },
        { "costmap",   "Costmap",    "/p_roboai_nav2/costmap",     QColor(255,160,0)   },
        { "scan",      "LaserScan",  "/amr/scan",                  QColor( 80,220,255) },
        { "path",      "Path",       "/p_roboai_nav2/path",        QColor( 50,230,100) },
        { "robot",     "Robot",      "/amr/odom",                  QColor( 60,180,255) },
        { "trail",     "Pose Trail", "/amr/odom",                  QColor(200,200, 80) },
        { "grid",      "Grid",       "(overlay)",                  QColor( 80, 80, 80) },
    };

    for (const auto& d : defs) {
        auto* sep = new QFrame(container);
        sep->setFrameShape(QFrame::HLine);
        sep->setStyleSheet("color:#3a3a3a;");

        auto* row = new DisplayRow(d.id, d.label, d.topic, d.color, container);
        connect(row, &DisplayRow::layerChanged, this, &DisplaysPanel::layerChanged);

        rows_layout->addWidget(sep);
        rows_layout->addWidget(row);
        _rows[d.id] = row;
    }
    rows_layout->addStretch();

    scroll->setWidget(container);
    outer->addWidget(title);
    outer->addWidget(scroll, 1);
}

bool DisplaysPanel::isLayerVisible(const QString& id) const
{
    auto it = _rows.find(id);
    return it != _rows.end() && it->second->isEnabled();
}

QColor DisplaysPanel::layerColor(const QString& id) const
{
    auto it = _rows.find(id);
    return it != _rows.end() ? it->second->color() : QColor(Qt::white);
}

float DisplaysPanel::layerOpacity(const QString& id) const
{
    auto it = _rows.find(id);
    return it != _rows.end() ? it->second->opacity() : 1.0f;
}

} // namespace p_roboai_viz
