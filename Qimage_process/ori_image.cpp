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

    setLED(ui.label_10, 2, 16);
    ui.label_11->setText("0");
    connect(ui.horizontalSlider, &QSlider::valueChanged, this, [this](int value) {
        ui.label_11->setText(QString::number(value));
    });
    connect(this, SIGNAL(camera_display(QImage, float, float, double)), this, SLOT(showImage(QImage, float, float, double)));
  
}

ori_image::~ori_image()
{}

void ori_image::showImage(QImage qimage,float width, float height, double elapsedSeconds) {
      
      qimage = qimage.scaled(ui.label->size(), Qt::KeepAspectRatio);
      QPixmap pixmap = QPixmap::fromImage(qimage);
      ui.label->setPixmap(pixmap);
      ui.label->show();  
      ui.label_14->setText(QString::number(width));
      ui.label_16->setText(QString::number(height));
      ui.label_6->setText(QString::number(elapsedSeconds));

}

void ori_image::camera_status(int arg1) {

    bool status = ui.checkBox->isChecked();
    if (status == true)
    {
        setLED(ui.label_10, 2, 16);

    }
    else if (status == false)
    {
        setLED(ui.label_10, 1, 16);
        //ui.label->close();
    }
}


void ori_image::setText(QString str){
    //ui.label->setText(str);
}

void ori_image::closed_view() {
    this->close();
}