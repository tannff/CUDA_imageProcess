#pragma once

#include <QtWidgets/QMainWindow>
#include "QDebug"
#include <iostream>
#include <QChart>   
#include <QBarSet> 
#include <QBarSeries> 
#include <QLineSeries>
#include <QLegend>
#include <QBarCategoryAxis>
#include <QValueAxis>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QDateTime>
#include <opencv2/opencv.hpp>
#include "ui_Qimage_process.h"
#include "ui_ori_image.h"
#include "ori_image.h"
#include "imageProcess.h"

class Qimage_process : public QMainWindow
{
    Q_OBJECT

public:
    Qimage_process(QWidget *parent = nullptr);
    ~Qimage_process();
    void setLED(QLabel* label, int color, int size);
    int resize_value;
    bool switch_resize;
    ori_image* ori_display;
    

signals:
    void image_proccess_speed(double, double, double, double, double, double, float, float, float, float, float, float);
    void width_measure(float, double);
    void image_resize(int, bool);
    void image_angle(float);
    void image_thresh(int);
    void image_demarcate(int);
    void image_ROI(int,int);
    void pre_image(QImage);


public slots:
    void display_speed(double, double, double, double, double, double, float, float, float, float, float, float);
    void display_width(float, double);
    void show_resize();
    void show_angle();
    void show_thresh();
    void show_demarcate();
    void show_ROI();
    void online_status();
    void offline_status();
    void show_the_file();
    void consumer();
    void new_window();
    void default_on();
    void save_file();
    void rotated_image(float);
    void return_image(QImage);
    void close_app();
    void gray_image();
    void thresh_image();

protected:
    void mouseMoveEvent(QMouseEvent* event);

private slots:
    void timer_Update(); //定时器更新槽函数
    void offline_image_down();

private:
    Ui::Qimage_processClass ui;
    QTimer* timer_calendar;
    QTimer* timer_unit;
    QString default_Path = "F:/about_the_lesson/Qimage_process/CUDA_imageProcess/";
    QImage image_ori;
    QImage q_image_gray;
    QImage q_image_thresh;
    QImage gray_cpu_image;
};
