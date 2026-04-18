#pragma once
#include <QCheckBox>
#include <QColor>
#include <QLabel>
#include <QScrollArea>
#include <QSlider>
#include <QString>
#include <QToolButton>
#include <QWidget>
#include <map>

namespace p_roboai_viz {

// ── Single display-layer row ──────────────────────────────────────────────────
class DisplayRow : public QWidget {
    Q_OBJECT
public:
    explicit DisplayRow(const QString& id,
                        const QString& label,
                        const QString& topic,
                        QColor         color,
                        QWidget*       parent = nullptr);

    bool    isEnabled()  const;
    QColor  color()      const { return _color; }
    float   opacity()    const;

signals:
    void layerChanged(const QString& id);

protected:
    bool eventFilter(QObject* obj, QEvent* ev) override;

private slots:
    void pickColor();

private:
    QString      _id;
    QColor       _color;
    QCheckBox*   _check{nullptr};
    QLabel*      _color_btn{nullptr};
    QSlider*     _opacity{nullptr};
};

// ── Panel containing all display rows ────────────────────────────────────────
class DisplaysPanel : public QWidget {
    Q_OBJECT
public:
    explicit DisplaysPanel(QWidget* parent = nullptr);

    bool   isLayerVisible(const QString& id) const;
    QColor layerColor    (const QString& id) const;
    float  layerOpacity  (const QString& id) const;

signals:
    void layerChanged(const QString& id);

private:
    void addRow(const QString& id,
                const QString& label,
                const QString& topic,
                QColor         color);

    std::map<QString, DisplayRow*> _rows;
};

} // namespace p_roboai_viz
