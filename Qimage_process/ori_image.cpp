#include "ori_image.h"
#include "camera_on.h"

using namespace cv;

void ori_image::setLED(QLabel* label, int color, int size)
{
    label->setText("");
    QString min_width = QString("min-width: %1px;").arg(size);             
    QString min_height = QString("min-height: %1px;").arg(size);           
    QString max_width = QString("max-width: %1px;").arg(size);             
    QString max_height = QString("max-height: %1px;").arg(size);            
    QString border_radius = QString("border-radius: %1px;").arg(size / 2);   
    QString border = QString("border:1px solid black;");                   
    QString background = "background-color:";
    switch (color) {
    case 0:
        // 灰色
        background += "rgb(190,190,190)";
        break;
    case 1:
        // 红色
        background += "rgb(255,0,0)";
        break;
    case 2:
        // 绿色
        background += "rgb(0,255,0)";
        break;
    case 3:
        // 黄色
        background += "rgb(255,255,0)";
        break;
    default:
        break;
    }
    const QString SheetStyle = min_width + min_height + max_width + max_height + border_radius + border + background;
    label->setStyleSheet(SheetStyle);
}

ori_image::ori_image(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

    setLED(ui.label_10, 1, 16);
    ui.label_11->setText("0");
    connect(ui.horizontalSlider, &QSlider::valueChanged, this, [this](int value) {
        ui.label_11->setText(QString::number(value));
    });
    connect(this, SIGNAL(camera_display(QImage)), this, SLOT(showImage(QImage)));
  
}

ori_image::~ori_image()
{}

void ori_image::showImage(QImage qimage) {
      
      qimage = qimage.scaled(ui.label->size(), Qt::KeepAspectRatio);
      QPixmap pixmap = QPixmap::fromImage(qimage);
      ui.label->setPixmap(pixmap);
      ui.label->show();  

}

void ori_image::camera_status() {
    setLED(ui.label_10, 2, 16);
}


void ori_image::setText(QString str){
    //ui.label->setText(str);
}

void ori_image::closed_view() {
    this->close();
}