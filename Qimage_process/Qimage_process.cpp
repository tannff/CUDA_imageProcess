#include "Qimage_process.h"
#include "camera_on.h"
#include "qtconcurrentrun.h"
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(ret)									\
	if (ret != cudaSuccess) {									\
		std::cerr << cudaGetErrorString(ret) << std::endl;		\
	}
extern cudaError_t rgb_to_gray(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hist);
extern cudaError_t gaussian_filter(unsigned char* img_in, unsigned char* img_gauss, int img_width, int img_height, int filterWidth, float* filter);
extern cudaError_t thresh_cal(const int* hist, float* sum, float* s, float* n, float* val, int img_width, int img_height, int* OtsuThresh);
extern cudaError_t gray_to_otsu_binary(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height, int* hThresh);
extern cudaError_t dilation(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height);
extern cudaError_t erosion(unsigned char* img_in, unsigned char* img_out, int img_width, int img_height);
extern cudaError_t sobel_intensity_gradient(unsigned char* img_in, unsigned char* img_sobel, int* Gx, int* Gy, int img_width, int img_height);
extern cudaError_t non_max(unsigned char* img_in, unsigned char* img_nms, int* Gx, int* Gy, int img_width, int img_height);
extern cudaError_t hysteresis(unsigned char* img_in, unsigned char* img_high, unsigned char* img_trace, unsigned* strong_edge_mask, int t_high, int t_low, int img_width, int img_height);

QT_CHARTS_USE_NAMESPACE      
using namespace std;
using namespace cv;

// 该函数将label控件变成一个圆形指示灯，需要指定颜色color以及直径size
// color 0:grey 1:red 2:green 3:yellow
// size  单位是像素
void Qimage_process::setLED(QLabel* label, int color, int size)
{
    // 将label中的文字清空
    label->setText("");
    // 先设置矩形大小
    // 如果ui界面设置的label大小比最小宽度和高度小，矩形将被设置为最小宽度和最小高度；
    // 如果ui界面设置的label大小比最小宽度和高度大，矩形将被设置为最大宽度和最大高度；
    QString min_width = QString("min-width: %1px;").arg(size);              // 最小宽度：size
    QString min_height = QString("min-height: %1px;").arg(size);            // 最小高度：size
    QString max_width = QString("max-width: %1px;").arg(size);              // 最小宽度：size
    QString max_height = QString("max-height: %1px;").arg(size);            // 最小高度：size
    // 再设置边界形状及边框
    QString border_radius = QString("border-radius: %1px;").arg(size / 2);    // 边框是圆角，半径为size/2
    QString border = QString("border:1px solid black;");                    // 边框为1px黑色
    // 最后设置背景颜色
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

Qimage_process::Qimage_process(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    ori_display = new ori_image;
    //实时时间显示
    timer_calendar = new QTimer(this);//new个对象
    connect(timer_calendar, SIGNAL(timeout()), this, SLOT(timer_Update()));//timeout超时事件
    timer_calendar->start(1000);//每隔一秒调用一次槽函数

    //图像显示界面初始化
    ui.label_6->setText(QString::fromLocal8Bit("原始图像"));
    ui.label_5->setText(QString::fromLocal8Bit("当前无图像！"));
    ui.label_2->setText(QString::fromLocal8Bit("采用CPU进行图像识别"));
    ui.label->setText(QString::fromLocal8Bit("当前无图像！"));
    ui.label_4->setText(QString::fromLocal8Bit("调用CUDA进行图像识别"));
    ui.label_3->setText(QString::fromLocal8Bit("当前无图像！"));

    //滑块初始化
    ui.label_46->setText("0");
    ui.label_47->setText("0");
    ui.label_48->setText("0");

    //默认保存地址
    ui.label_45->setText(default_Path);
    
    //距离变换结果显示
    ui.label_15->setText(QString::fromLocal8Bit("1毫米（mm）="));
    ui.label_28->setText(QString::fromLocal8Bit("像素（pixel)"));

    QGridLayout* baseLayout = new QGridLayout();//布局管理器

    connect(this, SIGNAL(image_proccess_speed(double, double, double, double, double, double, float, float, float, float, float, float)),
            this, SLOT(display_speed(double, double, double, double, double, double, float, float, float, float, float, float)));
    connect(this, SIGNAL(width_measure(float, double)), this, SLOT(display_width(float, double)));
    connect(ui.lineEdit_7, SIGNAL(returnPressed()), this, SLOT(show_resize()));
    connect(ui.lineEdit_6, SIGNAL(returnPressed()), this, SLOT(show_angle()));

    connect(ui.action, SIGNAL(triggered()), this, SLOT(offline_image_down()));
    connect(ui.action_8, SIGNAL(triggered()), this, SLOT(save_file()));
    connect(ui.action_4, SIGNAL(triggered()), this, SLOT(close_app()));


    //滑块设置
    connect(ui.verticalSlider, &QSlider::valueChanged, this, [this](int value) {    
        ui.label_46->setText(QString::number(value));
    });
    connect(ui.verticalSlider_3, &QSlider::valueChanged, this, [this](int value) {
        
        ui.label_47->setText(QString::number(value));
    });
    connect(ui.verticalSlider_2, &QSlider::valueChanged, this, [this](int value) {
        ui.label_48->setText(QString::number(value));
    });
    
    connect(ui.lineEdit_6, SIGNAL(returnPressed()), this, SLOT(rotated_image(float)));
    connect(ui.label_5, SIGNAL(pre_image(QImage)), this, SLOT(return_image(QImage)));

    //设置默认值
    ui.lineEdit->setPlaceholderText("40");
    ui.lineEdit_2->setPlaceholderText("480");
    ui.lineEdit_3->setPlaceholderText("320");
    ui.lineEdit_5->setPlaceholderText("63");
    ui.lineEdit_6->setPlaceholderText("0");
    ui.lineEdit_7->setPlaceholderText("100");

    //设置状态灯初态
    setLED(ui.label_9, 1, 16);
    setLED(ui.label_40, 2, 16);
}

Qimage_process::~Qimage_process()
{

    
}

void Qimage_process::timer_Update()
{
    QDateTime time = QDateTime::currentDateTime();
    //年-月-日 时：分：秒 星期
    QString str = time.toString("yyyy-MM-dd hh:mm:ss dddd");
    ui.label_43->setText(str);
}


void Qimage_process::display_speed(double times1, double times2, double times3, double times4, double times5, double times6, float cost_time, float gtimes1, float gtimes2, float gtimes3, float gtimes4, float gtimes5) {
    
    //构建折线图

    QLineSeries* series1 = new QLineSeries();
    QLineSeries* series2 = new QLineSeries();

    series1->setName("cpu");
    series1->append(0, 0);
    series1->append(times1, 1);
    series1->append(times1 + times2, 2);
    series1->append(times1 + times2 + times3, 3);
    series1->append(times1 + times2 + times3 + times4, 4);
    series1->append(times1 + times2 + times3 + times4 + times5, 5);
    series1->append(times1 + times2 + times3 + times4 + times5 + times6, 6);

    series2->setName("cuda");
    series2->append(0, 0);
    series2->append(gtimes1, 1);
    series2->append(gtimes1 + gtimes2, 2);
    series2->append(gtimes1 + gtimes2 + gtimes3, 3);
    series2->append(gtimes1 + gtimes2 + gtimes3 + gtimes4, 4);
    series2->append(gtimes1 + gtimes2 + gtimes3 + gtimes4 + gtimes5, 5);
    series2->append(1+gtimes1 + gtimes2 + gtimes3 + gtimes4 + gtimes5, 6);

    QChart* chart = new QChart();
    chart->addSeries(series1);
    chart->addSeries(series2);

    //chart->legend()->hide();
    
    chart->createDefaultAxes();
    chart->legend()->setVisible(true); //设置图例为显示状态
    chart->legend()->setAlignment(Qt::AlignLeft);//设置图例的显示位置在底部
    chart->legend()->setContentsMargins(5, 0, 0, 0);  

    QChartView* chartView = new QChartView();
    ui.graphicsView->setChart(chart);

    QValueAxis* axisX = new QValueAxis;  //X轴
    axisX->setRange(0, 50);   //设置坐标轴范围
    axisX->setTitleText("time(ms)");  //标题
    axisX->setLabelFormat("%.2f");   //标签格式
    axisX->setTickCount(5);    //主分隔个数
    axisX->setMinorTickCount(1);    //每个单位之间绘制了多少虚网线
    chart->setAxisX(axisX, series1);
    chart->setAxisX(axisX, series2);

    ui.graphicsView->setRenderHint(QPainter::Antialiasing);
    ui.graphicsView->setRubberBand(QChartView::NoRubberBand);

    //构建柱状图
    QBarSet* set0 = new QBarSet("cpu");
    QBarSet* set1 = new QBarSet("cuda");

    *set0 << times1 << times2 << times3 << times4 << times5 << times6;
    *set1 << gtimes1 << gtimes2 << gtimes3 << gtimes4 << gtimes5 << 0;

    QBarSeries* series = new QBarSeries();
    series->append(set0);
    series->append(set1);

    QChart* chartc = new QChart();
    ui.graphicsView_2->setChart(chartc);

    QValueAxis* axisY = new QValueAxis;  //Y轴
    axisY->setRange(0, 18);
    axisY->setTitleText("time(ms)");
    axisY->setLabelFormat("%.1f"); 
    axisY->setTickCount(5); 
    axisY->setMinorTickCount(1); 
    chartc->setAxisY(axisY, series);

    chartc->addSeries(series);

    chartc->setTitle(QString::fromLocal8Bit("cpu与cuda图像处理时间对比"));
    chartc->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << QString::fromLocal8Bit("灰度化")
               << QString::fromLocal8Bit("阈值化")
               << QString::fromLocal8Bit("滤波")
               << QString::fromLocal8Bit("形态学")
               << QString::fromLocal8Bit("边缘检测")
               << QString::fromLocal8Bit("距离变换");

    QBarCategoryAxis* axis = new QBarCategoryAxis();
    axis->append(categories);
    QFont font;
    font.setPointSize(6);  // 设置字体大小
    axis->setLabelsFont(font);
    //chartc->createDefaultAxes();//创建默认的左侧的坐标轴（根据 QBarSet 设置的值）
    
    chartc->setAxisX(axis, series);//设置坐标轴

    chartc->legend()->setVisible(false); //设置图例为显示状态
    chartc->legend()->setAlignment(Qt::AlignRight);//设置图例的显示位置在底部

    /*QChartView* chartView = new QChartView(chartc);
    chartView = new QChartView();
    baseLayout->addWidget(chartView, 1, 0);
    setLayout(baseLayout);//Qwidget特有的*/

    ui.graphicsView_2->setRenderHint(QPainter::Antialiasing);
    ui.graphicsView_2->setRubberBand(QChartView::NoRubberBand);
     
}

void Qimage_process::show_resize() {

    QString re = ui.lineEdit_7->text();
    bool ok;
    int resize_value = re.toInt(&ok,10);
    image_ori = image_ori.scaled(resize_value * image_ori.width()/100, resize_value * image_ori.height()/100, Qt::KeepAspectRatio);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    ui.pushButton->setChecked(false);
    ui.pushButton_2->setChecked(false);
    ui.pushButton_3->setChecked(false);
    ui.pushButton_4->setChecked(false);
    ui.pushButton_5->setChecked(false);
    ui.pushButton_6->setChecked(false);
    ui.pushButton_7->setChecked(false);
    ui.pushButton_8->setChecked(false);
}

void Qimage_process::show_angle() {
    QString an = ui.lineEdit_6->text();
    bool ok;
    float angle_value = an.toFloat(&ok);
    if (angle_value < 0.0 || angle_value > 360.0) {
        return;
    }
    QMatrix matrix;
    matrix.rotate(angle_value);
    //QImage rotated_image(image_ori.size(), image_ori.format());
    image_ori = image_ori.transformed(matrix, Qt::FastTransformation);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    ui.pushButton->setChecked(false);
    ui.pushButton_2->setChecked(false);
    ui.pushButton_3->setChecked(false);
    ui.pushButton_4->setChecked(false);
    ui.pushButton_5->setChecked(false);
    ui.pushButton_6->setChecked(false);
    ui.pushButton_7->setChecked(false);
    ui.pushButton_8->setChecked(false);
}

void Qimage_process::horizenal_flip() {
        image_ori = image_ori.mirrored(true, false);
        image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
        ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

        ui.pushButton->setChecked(false);
        ui.pushButton_2->setChecked(false);
        ui.pushButton_3->setChecked(false);
        ui.pushButton_4->setChecked(false);
        ui.pushButton_5->setChecked(false);
        ui.pushButton_6->setChecked(false);
        ui.pushButton_7->setChecked(false);
        ui.pushButton_8->setChecked(false);
}

void Qimage_process::vertical_flip() {
    image_ori = image_ori.mirrored(false, true);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    ui.pushButton->setChecked(false);
    ui.pushButton_2->setChecked(false);
    ui.pushButton_3->setChecked(false);
    ui.pushButton_4->setChecked(false);
    ui.pushButton_5->setChecked(false);
    ui.pushButton_6->setChecked(false);
    ui.pushButton_7->setChecked(false);
    ui.pushButton_8->setChecked(false);
}

void Qimage_process::show_thresh() {
    QString th = ui.lineEdit->text();
    bool ok;
    int thresh_value = th.toInt(&ok,10);
    if (thresh_value < 0 || thresh_value > 255) {
        return;
    }
    emit image_thresh(thresh_value);
}

void Qimage_process::show_demarcate() {
   
    QString de = ui.lineEdit_5->text();
    bool ok;
    int demarcate_value = de.toInt(&ok, 10);
    
    emit image_demarcate(demarcate_value);


}

void Qimage_process::show_ROI() {
    QString Rw = ui.lineEdit_2->text();
    QString Rh = ui.lineEdit_3->text();
    bool ok;
    int ROI_width = Rw.toInt(&ok, 10);
    int ROI_height = Rh.toInt(&ok, 10);
    if (ROI_width < 0 || ROI_width > 1920) {
        if (ROI_height < 0 || ROI_height > 1080) {
            return;
        }
    }
    image_ori = image_ori.scaled(ROI_width, ROI_height, Qt::IgnoreAspectRatio);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));
}

void Qimage_process::online_status() {
    setLED(ui.label_9, 2, 16);
    setLED(ui.label_40, 1, 16);
}

void Qimage_process::offline_status() {
    setLED(ui.label_9, 1, 16);
    setLED(ui.label_40, 2, 16);
}

void Qimage_process::show_the_file() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("open a file."),
        "F:/about_the_lesson/Qimage_process/CUDA_imageProcess/",
        tr("images(*.png *jpg *bmp);;video files(*.avi *.mp4 *.wmv);;All files(*.*)"));
        
    if (fileName.isEmpty()) {
        QMessageBox::warning(this, "Warning!", "Failed to open the image!");
    }
    image_ori = QImage(fileName);
    image_ori = image_ori.scaled(ui.label->size(), Qt::KeepAspectRatio);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    QPixmap pix_image_ori = QPixmap::fromImage(image_ori);
    ui.label_5->setPixmap(pix_image_ori);  
    emit pre_image(image_ori);

    ui.pushButton->setChecked(false);
    ui.pushButton_2->setChecked(false);
    ui.pushButton_3->setChecked(false);
    ui.pushButton_4->setChecked(false);
    ui.pushButton_5->setChecked(false);
    ui.pushButton_6->setChecked(false);
    ui.pushButton_7->setChecked(false);
    ui.pushButton_8->setChecked(false);

}

void Qimage_process::display_width(float maxValue, double interpolation) {
    
    float mm_elipse_width = (maxValue * 2) / 63.00;
    float mil_width = mm_elipse_width / 0.0254;
    float mil_interpolation = (double)interpolation / 0.0254;

    if (ui.comboBox_5->currentText() == "mm")
    {
        ui.label_19->setText(QString::number(mm_elipse_width));
        //ui.label_20->setText(QString::number());

    }
    else if (ui.comboBox_5->currentText() == "mil")
    {
        ui.label_19->setText(QString::number(mil_width));
        //ui.label_20->setText(QString::number());
    }

    if (ui.comboBox_7->currentText() == "mm")
    {
        ui.label_21->setText(QString::number(interpolation));
        //ui.label_22->setText(QString::number());

    }
    else if (ui.comboBox_7->currentText() == "mil")
    {
        ui.label_21->setText(QString::number(mil_interpolation));
    }
}

void Qimage_process::consumer() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("open a file."),
        "F:/about_the_lesson/",
        tr("consumer_file(*.cpp *h *cu);;All files(*.*)"));
        QFileInfo fileInfo(fileName);
        QString baseName = fileInfo.fileName();

        // 将文件名显示在标签上
        ui.label_42->setText(baseName);

}

void Qimage_process::new_window() {

    ori_display->show();
    QtConcurrent::run(camera_on,ori_display);
    
    ori_display->setText(ui.label_42->text());  
}

void Qimage_process::offline_image_down() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        tr("open a file."),
        "F:/about_the_lesson/Qimage_process/CUDA_imageProcess/",
        tr("images(*.png *jpg *bmp);;video files(*.avi *.mp4 *.wmv);;All files(*.*)"));

}

void Qimage_process::mouseMoveEvent(QMouseEvent* event)
{
    //ui.label_44->setText(QString::number(event->x()));//以窗口左上角为0点
    if (event->button() == Qt::LeftButton) {
        // 获取鼠标点击的相对坐标
        QPoint pos = mapFromGlobal(QCursor::pos());
        int x = pos.x();
        int y = pos.y();
        ui.label_44->setText(QString::number(x));
        //ui.label_44->setText(QString::number(event->x()));

    }

        //// 获取图像像素值
        //QImage image = pixmap.toImage(); // pixmap是一个QPixmap对象
        //QRgb pixel = image.pixel(x, y);
        //int r = qRed(pixel);
        //int g = qGreen(pixel);
        //int b = qBlue(pixel);
        //qDebug() << "Pixel value at (" << x << "," << y << "): (" << r << "," << g << "," << b << ")";

    

    //    ui->showX->setText(QString::number(QCursor().pos().x()));//以电脑屏幕左上角为0点
    //    ui->showY->setText(QString::number(QCursor().pos().y()));

    //  ui->showdata->setText(tr("(%1,%2)").arg(event->x()).arg(event->y())); //哦。。一个标签显示两个变量是这么搞的 不好意思丢人了
    //arg()是QString类中的一个静态函数，使用它就可以在字符串中使用变量了。所以就不用那个强制类型转换了
    //QColor pixcolor = QColor(pool.pixel(event->x(), event->y()));//有pool.pixelColor()不知道干嘛的
   /* ui->showR->setText("R" + QString::number(pixcolor.red()));
    ui->showG->setText("G" + QString::number(pixcolor.green()));
    ui->showB->setText("B" + QString::number(pixcolor.blue()));*/

}

void Qimage_process::default_on(){
    ui.pushButton_23->setDefault(true);
}

void Qimage_process::save_file() {

    QString fileName = QFileDialog::getSaveFileName(
        this,
        tr("save a file."),
        default_Path,
        tr("images(*.png *jpg *bmp);;video files(*.avi *.mp4 *.wmv);;All files(*.*)"));

        ui.label_45->setText(fileName.isEmpty() ? default_Path : fileName);
}

void Qimage_process::return_image() {
    
    QPixmap pix_image_ori = QPixmap::fromImage(image_ori);
    ui.label_5->setPixmap(pix_image_ori);
}

void Qimage_process::close_app() {
    this->close();
}

void Qimage_process::gray_image(bool gray_on) {
    if (gray_on) {
        //CUDA:to gray_image
        unsigned char* d_rgb;
        
        qDebug() << image_ori;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rgb, image_ori.width() * image_ori.height() * image_ori.depth() / 8 * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_rgb, image_ori.bits(), image_ori.width() * image_ori.height() * image_ori.depth() / 8 * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_gray;
        cudaMalloc((void**)&d_gray, image_ori.width() * image_ori.height() * sizeof(unsigned char));

        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_hist, 256 * sizeof(int)));
        CHECK_CUDA_ERROR(rgb_to_gray(d_rgb, d_gray, image_ori.width(), image_ori.height(), d_hist));

        q_image_gray = QImage(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_gray.bits(), d_gray, image_ori.width() * image_ori.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_gray));

        CHECK_CUDA_ERROR(cudaFree(d_rgb));
        CHECK_CUDA_ERROR(cudaFree(d_gray));

        //cpu:to gray_image
        gray_cpu_image = QImage(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
        rgb2grayincpu(image_ori.bits(), gray_cpu_image.bits(), image_ori.width(), image_ori.height());
        ui.label->setPixmap(QPixmap::fromImage(gray_cpu_image));
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(image_ori));
        ui.label->setPixmap(QPixmap::fromImage(image_ori));
    }
}

void Qimage_process::filter_image(bool filter_on) {


    float Sigma = 1;
    int filterWidth = 5;
    if (filterWidth < 3) filterWidth = 3;
    else filterWidth = (int)(filterWidth / 2) * 2 + 1;
    float* filter = new float[filterWidth * filterWidth];   //生成高斯核
    int center = filterWidth / 2;
    float sum = 0;

    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            filter[i * filterWidth + j] = exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * Sigma * Sigma));
            sum += filter[i * filterWidth + j];
        }
    }
    double sum1 = 1 / sum;
    for (int i = 0; i < filterWidth; i++)
    {
        for (int j = 0; j < filterWidth; j++)
        {
            filter[i * filterWidth + j] *= sum1;  //高斯卷积核归一化
        }
    }

    if (filter_on) {
        //CUDA:to filter_image
        ui.label_3->clear();
        ui.label_3->setPixmap(QPixmap());

        unsigned char* d_gray;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gray, q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gray, q_image_gray.bits(), q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_gauss;
        cudaMalloc((void**)&d_gauss, q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char));

        float* d_filter;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_filter, filterWidth * filterWidth * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERROR(gaussian_filter(d_gray, d_gauss, q_image_gray.width(), q_image_gray.height(), filterWidth, d_filter));

        q_image_gauss = QImage(q_image_gray.width(), q_image_gray.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_gauss.bits(), d_gauss, q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_gauss));
        QCoreApplication::processEvents();

        CHECK_CUDA_ERROR(cudaFree(d_gray));
        CHECK_CUDA_ERROR(cudaFree(d_gauss));
        CHECK_CUDA_ERROR(cudaFree(d_filter));

        //cpu:to filter_image
        cv::Mat gray_mat = cv::Mat(gray_cpu_image.height(), gray_cpu_image.width(), CV_8UC1, (void*)gray_cpu_image.bits(), gray_cpu_image.bytesPerLine());
        cv::Mat filter_mat(gray_mat.cols, gray_mat.rows, CV_8UC1, Scalar(0));
        GaussianBlur(gray_mat, filter_mat, Size(5, 5), 0, 0);
        filter_cpu_image = QImage(filter_mat.data, filter_mat.cols, filter_mat.rows, QImage::Format_Grayscale8);
        
        ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_gray));
        ui.label->setPixmap(QPixmap::fromImage(gray_cpu_image));
    }
}

void Qimage_process::thresh_image(bool binary_on) {
    if (binary_on) {
        //CUDA:to thresh_image
        ui.label_3->clear();

        ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
        unsigned char* d_gauss;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gauss, q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gauss, q_image_gauss.bits(), q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_thresh;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_thresh, q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char)));

        float* d_sum;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum, 256 * sizeof(int)));
        float* d_s;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_s, sizeof(float)));
        float* d_n;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_n, 256 * sizeof(float)));
        float* d_val;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_val, 256 * sizeof(float)));
        int* d_t;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t, 2 * sizeof(int)));

        CHECK_CUDA_ERROR(thresh_cal(d_hist, d_sum, d_s, d_n, d_val, q_image_gauss.width(), q_image_gauss.height(), d_t));
        CHECK_CUDA_ERROR(gray_to_otsu_binary(d_gauss, d_thresh, q_image_gauss.width(), q_image_gauss.height(), d_t));

        q_image_thresh = QImage(q_image_gauss.width(), q_image_gauss.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_thresh.bits(), d_thresh, q_image_thresh.width() * q_image_thresh.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_thresh));
        QCoreApplication::processEvents();

        CHECK_CUDA_ERROR(cudaFree(d_gauss));
        CHECK_CUDA_ERROR(cudaFree(d_hist));
        CHECK_CUDA_ERROR(cudaFree(d_sum));
        CHECK_CUDA_ERROR(cudaFree(d_s));
        CHECK_CUDA_ERROR(cudaFree(d_n));
        CHECK_CUDA_ERROR(cudaFree(d_val));
        CHECK_CUDA_ERROR(cudaFree(d_t));
        CHECK_CUDA_ERROR(cudaFree(d_thresh));

        //cpu:to thresh_image
        //ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
        //cv::Mat filter_mat = cv::Mat(filter_cpu_image.height(), filter_cpu_image.width(), CV_8UC1,  (void*)filter_cpu_image.bits(), filter_cpu_image.bytesPerLine());
        //cv::Mat binary_mat(filter_mat.cols, filter_mat.rows, CV_8UC1);
        //imshow("", filter_mat);
        //threshold(filter_mat, binary_mat, 40, 255, THRESH_BINARY /*| THRESH_OTSU*/);
        //binary_cpu_image = QImage(binary_mat.data, binary_mat.cols, binary_mat.rows, QImage::Format_Grayscale8);

       // ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_gauss));
        ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
    }
}

void Qimage_process::closed_image(bool closed_on) {
    if (closed_on) {
        //CUDA:to closed_image
        ui.label_3->clear();
        ui.label_3->setPixmap(QPixmap());

        unsigned char* d_thresh;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_thresh, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_thresh, q_image_thresh.bits(), image_ori.width() * image_ori.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_dil;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dil, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
        unsigned char* d_closed;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, image_ori.width() * image_ori.height() * sizeof(unsigned char)));

        CHECK_CUDA_ERROR(dilation(d_thresh, d_dil, image_ori.width(), image_ori.height()));
        CHECK_CUDA_ERROR(erosion(d_dil, d_closed, image_ori.width(), image_ori.height()));

        q_image_closed = QImage(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_closed.bits(), d_closed, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_closed));
        QCoreApplication::processEvents();

        CHECK_CUDA_ERROR(cudaFree(d_dil));
        CHECK_CUDA_ERROR(cudaFree(d_closed));

        //cpu:to closed_image
        //QImage gray_cpu_image(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
        //rgb2grayincpu(image_ori.bits(), gray_cpu_image.bits(), image_ori.width(), image_ori.height());

        //cv::Mat gray_mat = cv::Mat(gray_cpu_image.height(), gray_cpu_image.width(), CV_8UC1, (void*)gray_cpu_image.bits(), gray_cpu_image.bytesPerLine());
        //cv::Mat filter_mat(gray_mat.cols, gray_mat.rows, CV_8UC1);
        //GaussianBlur(gray_mat, filter_mat, Size(5, 5), 0, 0);

        //cv::Mat binary_mat(filter_mat.cols, filter_mat.rows, CV_8UC1);
        //threshold(filter_mat, binary_mat, 45, 255, THRESH_BINARY /*| THRESH_OTSU*/);
        cv::Mat binary_mat(binary_cpu_image.height(), binary_cpu_image.width(), CV_8UC1, (void*)binary_cpu_image.bits(), binary_cpu_image.bytesPerLine());
        cv::Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
        cv::Mat closed_mat(binary_mat.size(), CV_8UC1, Scalar(0));
        morphologyEx(binary_mat, closed_mat, MORPH_CLOSE, element);
        closed_cpu_image = QImage(closed_mat.data, closed_mat.cols, closed_mat.rows, QImage::Format_Grayscale8);

        ui.label->setPixmap(QPixmap::fromImage(closed_cpu_image));
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_thresh));
        QImage gray_cpu_image(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
        rgb2grayincpu(image_ori.bits(), gray_cpu_image.bits(), image_ori.width(), image_ori.height());

        cv::Mat gray_mat = cv::Mat(gray_cpu_image.height(), gray_cpu_image.width(), CV_8UC1, (void*)gray_cpu_image.bits(), gray_cpu_image.bytesPerLine());
        cv::Mat filter_mat(gray_mat.cols, gray_mat.rows, CV_8UC1);
        GaussianBlur(gray_mat, filter_mat, Size(5, 5), 0, 0);

        cv::Mat binary_mat(filter_mat.cols, filter_mat.rows, CV_8UC1);
        threshold(filter_mat, binary_mat, 45, 255, THRESH_BINARY /*| THRESH_OTSU*/);
        binary_cpu_image = QImage(binary_mat.data, binary_mat.cols, binary_mat.rows, QImage::Format_Grayscale8);

        ui.label->setPixmap(QPixmap::fromImage(binary_cpu_image));
    }
}

void Qimage_process::canny_image() {
    //定义阈值
    int t_high = 150;
    int t_low = 50;
    //CUDA:to cann_image
    ui.label_3->clear();
    ui.label_3->setPixmap(QPixmap());

    unsigned char* d_closed;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_closed, q_image_closed.bits(), image_ori.width() * image_ori.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    unsigned char* d_sobel;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sobel, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
    unsigned char* d_nms;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nms, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
    unsigned char* d_high;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_high, image_ori.width() * image_ori.height() * sizeof(unsigned char)));
    unsigned char* d_canny;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_canny, image_ori.width() * image_ori.height() * sizeof(unsigned char)));

    int* d_gx;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gx, image_ori.width() * image_ori.height() * sizeof(int)));
    int* d_gy;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gy, image_ori.width() * image_ori.height() * sizeof(int)));
    unsigned* d_map;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_map, image_ori.width() * image_ori.height() * sizeof(d_map[0])));

    CHECK_CUDA_ERROR(sobel_intensity_gradient(d_closed, d_sobel, d_gx, d_gy, image_ori.width(), image_ori.height()));
    CHECK_CUDA_ERROR(non_max(d_sobel, d_nms, d_gx, d_gy, image_ori.width(), image_ori.height()));
    CHECK_CUDA_ERROR(hysteresis(d_nms, d_high, d_canny, d_map, t_high, t_low, image_ori.width(), image_ori.height()));

    q_image_canny = QImage(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
    CHECK_CUDA_ERROR(cudaMemcpy(q_image_canny.bits(), d_canny, q_image_canny.width() * q_image_canny.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    ui.label_3->setPixmap(QPixmap::fromImage(q_image_canny));
    QCoreApplication::processEvents();

    CHECK_CUDA_ERROR(cudaFree(d_closed));
    CHECK_CUDA_ERROR(cudaFree(d_sobel));
    CHECK_CUDA_ERROR(cudaFree(d_nms));
    CHECK_CUDA_ERROR(cudaFree(d_gx));
    CHECK_CUDA_ERROR(cudaFree(d_gy));
    CHECK_CUDA_ERROR(cudaFree(d_high));
    CHECK_CUDA_ERROR(cudaFree(d_canny));

    //cpu:to canny_image
    QImage gray_cpu_image(image_ori.width(), image_ori.height(), QImage::Format_Grayscale8);
    rgb2grayincpu(image_ori.bits(), gray_cpu_image.bits(), image_ori.width(), image_ori.height());

    cv::Mat gray_mat = cv::Mat(gray_cpu_image.height(), gray_cpu_image.width(), CV_8UC1, (void*)gray_cpu_image.bits(), gray_cpu_image.bytesPerLine());
    cv::Mat binary_mat(gray_mat.cols, gray_mat.rows, CV_8UC1);
    threshold(gray_mat, binary_mat, 45, 255, THRESH_BINARY /*| THRESH_OTSU*/);
    //QImage binary_cpu_image(binary_mat.data, binary_mat.cols, binary_mat.rows, QImage::Format_Grayscale8);

    cv::Mat filter_mat(binary_mat.cols, binary_mat.rows, CV_8UC1);
    GaussianBlur(binary_mat, filter_mat, Size(5, 5), 0, 0);
    //QImage filter_cpu_image(filter_mat.data, filter_mat.cols, filter_mat.rows, QImage::Format_Grayscale8);

    cv::Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    cv::Mat closed_mat(filter_mat.size(), CV_8UC1, Scalar(0));
    morphologyEx(filter_mat, closed_mat, MORPH_CLOSE, element);
    //QImage closed_cpu_image(closed_mat.data, closed_mat.cols, closed_mat.rows, QImage::Format_Grayscale8);

    //cv::Mat closed_mat(closed_cpu_image.height(), closed_cpu_image.width(), CV_8UC1, (void*)closed_cpu_image.bits(), closed_cpu_image.bytesPerLine());
    cv::Mat canny_mat(closed_mat.cols, closed_mat.rows, CV_8U, Scalar(0));
    Canny(closed_mat, canny_mat, t_low, t_high);
    vector<vector<Point> > contour_vec;
    vector<Vec4i> hierarchy;
    findContours(canny_mat, contour_vec, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    cv::Mat ori_image_c = cv::Mat(image_ori.height(), image_ori.width(), CV_8UC3, (void*)image_ori.bits(), image_ori.bytesPerLine());
    for (int real_contour = 0; real_contour < contour_vec.size(); ++real_contour) {
        drawContours(ori_image_c, contour_vec, real_contour, Scalar(255, 255, 255), 2, 8);
    }
    QImage canny_cpu_image(canny_mat.data, canny_mat.cols, canny_mat.rows, QImage::Format_Grayscale8);

    ui.label->setPixmap(QPixmap::fromImage(canny_cpu_image));
}