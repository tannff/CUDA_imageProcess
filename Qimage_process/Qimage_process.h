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
    void init_pushButton();
    ori_image* ori_display;
    

signals:
    void image_proccess_speed(double, double, double, double, double, double, float, float, float, float, float, float);
    void width_measure(float, double);
    void image_resize(int, bool);
    void image_angle(float);
    void image_thresh(int);
    void image_demarcate(int);
    void image_ROI(int,int);


public slots:
    void display_speed(double, double, double, double, double, double, float, float, float, float, float, float);
    void display_width(float, double);
    void show_resize();
    void show_angle();
    void show_demarcate();
    void show_ROI();
    void online_status();
    void offline_status();
    void show_the_file();
    void consumer();
    void new_window();
    void default_on();
    void save_file();
    void return_image();
    void close_app();
    void horizenal_flip();
    void vertical_flip();
    void paint(QString);
    void renew(int);
    void saturation(int);
    void gray_image(bool);
    void thresh_image(bool);
    void filter_image(bool);
    void closed_image(bool);
    void canny_image(bool);
    void distancetransform(bool);

protected:
    void mouseMoveEvent(QMouseEvent* event);

private slots:
    void timer_Update(); //定时器更新槽函数

private:
    Ui::Qimage_processClass ui;
    QTimer* timer_calendar;
    QTimer* timer_unit;
    QString default_Path = "F:/about_the_lesson/Qimage_process/CUDA_imageProcess/";
    QImage image_ori;
    QImage image_ori_copy;
    QImage q_image_gray;
    QImage q_image_thresh;
    QImage q_image_gauss;
    QImage q_image_closed;
    QImage q_image_canny;
    QImage q_image_distrans;
    QImage gray_cpu_image;
    QImage filter_cpu_image;
    QImage binary_cpu_image;
    QImage closed_cpu_image;
    QImage canny_cpu_image;
    QImage dist_cpu_image;
    int* d_hist;
};
