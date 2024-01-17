#pragma once

#include <QWidget>
#include "ui_ori_image.h"

namespace Ui {
	class ori_image;
}

class ori_image : public QWidget
{
	Q_OBJECT

public:
	ori_image(QWidget *parent = nullptr);
	~ori_image();
	void setLED(QLabel* label, int color, int size);
	void setText(QString str);

signals:
	void camera_display(QImage);

public slots:
	void camera_status();
	void showImage(QImage);
	void closed_view();

private:
	Ui::ori_imageClass ui;
};

