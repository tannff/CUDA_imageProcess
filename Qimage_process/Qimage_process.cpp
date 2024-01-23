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
extern cudaError_t distance_transform(unsigned char* img_in, unsigned char* img_out, const int img_width, const int img_height, int* max_x, int* max_y);

QT_CHARTS_USE_NAMESPACE      
using namespace std;
using namespace cv;

void Qimage_process::init_pushButton() {
    ui.pushButton->setChecked(false);
    ui.pushButton_2->setChecked(false);
    ui.pushButton_3->setChecked(false);
    ui.pushButton_4->setChecked(false);
    ui.pushButton_5->setChecked(false);
    ui.pushButton_6->setChecked(false);
    ui.pushButton_7->setChecked(false);
    ui.pushButton_8->setChecked(false);
}

void Qimage_process::setLED(QLabel* label, int color, int size)
{
    // ï¿½ï¿½labelï¿½Ðµï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
    label->setText("");
    // ï¿½ï¿½ï¿½ï¿½ï¿½Ã¾ï¿½ï¿½Î´ï¿½Ð¡
    // ï¿½ï¿½ï¿½uiï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ãµï¿½labelï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ÈºÍ¸ß¶ï¿½Ð¡ï¿½ï¿½ï¿½ï¿½ï¿½Î½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½Ð¡ï¿½ï¿½ï¿½Èºï¿½ï¿½ï¿½Ð¡ï¿½ß¶È£ï¿½
    // ï¿½ï¿½ï¿½uiï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ãµï¿½labelï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ÈºÍ¸ß¶È´ó£¬¾ï¿½ï¿½Î½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Îªï¿½ï¿½ï¿½ï¿½ï¿½Èºï¿½ï¿½ï¿½ï¿½ß¶È£ï¿½
    QString min_width = QString("min-width: %1px;").arg(size);              // ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½È£ï¿½size
    QString min_height = QString("min-height: %1px;").arg(size);            // ï¿½ï¿½Ð¡ï¿½ß¶È£ï¿½size
    QString max_width = QString("max-width: %1px;").arg(size);              // ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½È£ï¿½size
    QString max_height = QString("max-height: %1px;").arg(size);            // ï¿½ï¿½Ð¡ï¿½ß¶È£ï¿½size
    // ï¿½ï¿½ï¿½ï¿½ï¿½Ã±ß½ï¿½ï¿½ï¿½×´ï¿½ï¿½ï¿½ß¿ï¿½
    QString border_radius = QString("border-radius: %1px;").arg(size / 2);    // ï¿½ß¿ï¿½ï¿½ï¿½Ô²ï¿½Ç£ï¿½ï¿½ë¾¶Îªsize/2
    QString border = QString("border:1px solid black;");                    // ï¿½ß¿ï¿½Îª1pxï¿½ï¿½É«
    // ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ã±ï¿½ï¿½ï¿½ï¿½ï¿½É?
    QString background = "background-color:";
    switch (color) {
    case 0:
        // ï¿½ï¿½É«
        background += "rgb(190,190,190)";
        break;
    case 1:
        // ï¿½ï¿½É«
        background += "rgb(255,0,0)";
        break;
    case 2:
        // ï¿½ï¿½É«
        background += "rgb(0,255,0)";
        break;
    case 3:
        // ï¿½ï¿½É«
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
    //ÊµÊ±Ê±ï¿½ï¿½ï¿½ï¿½Ê¾
    timer_calendar = new QTimer(this);//newï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
    connect(timer_calendar, SIGNAL(timeout()), this, SLOT(timer_Update()));//timeoutï¿½ï¿½Ê±ï¿½Â¼ï¿½
    timer_calendar->start(1000);//Ã¿ï¿½ï¿½Ò»ï¿½ï¿½ï¿½ï¿½ï¿½Ò»ï¿½Î²Ûºï¿½ï¿½ï¿?

    //Í¼ï¿½ï¿½ï¿½ï¿½Ê¾ï¿½ï¿½ï¿½ï¿½ï¿½Ê¼ï¿½ï¿?
    ui.label_6->setText(QString::fromLocal8Bit("Ô­Ê¼Í¼ï¿½ï¿½"));
    ui.label_5->setText(QString::fromLocal8Bit("ï¿½ï¿½Ç°ï¿½ï¿½Í¼ï¿½ï¿½"));
    ui.label_2->setText(QString::fromLocal8Bit("ï¿½ï¿½ï¿½ï¿½CPUï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½Ê¶ï¿½ï¿½"));
    ui.label->setText(QString::fromLocal8Bit("ï¿½ï¿½Ç°ï¿½ï¿½Í¼ï¿½ï¿½"));
    ui.label_4->setText(QString::fromLocal8Bit("ï¿½ï¿½ï¿½ï¿½CUDAï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½Ê¶ï¿½ï¿½"));
    ui.label_3->setText(QString::fromLocal8Bit("ï¿½ï¿½Ç°ï¿½ï¿½Í¼ï¿½ï¿½"));

    //ï¿½ï¿½ï¿½ï¿½ï¿½Ê¼ï¿½ï¿?
    ui.label_46->setText("0");
    ui.label_47->setText("0");
    ui.label_48->setText("0");

    //Ä¬ï¿½Ï±ï¿½ï¿½ï¿½ï¿½Ö?
    ui.label_45->setText(default_Path);
    
    //ï¿½ï¿½ï¿½ï¿½ä»»ï¿½ï¿½ï¿½ï¿½ï¿½Ê¾
    ui.label_15->setText(QString::fromLocal8Bit("1ï¿½ï¿½ï¿½×£ï¿½mmï¿½ï¿½="));
    ui.label_28->setText(QString::fromLocal8Bit("ï¿½ï¿½ï¿½Ø£ï¿½pixel)"));

    connect(this, SIGNAL(image_proccess_speed(double, double, double, double, double, double, float, float, float, float, float, float)),
            this, SLOT(display_speed(double, double, double, double, double, double, float, float, float, float, float, float)));
    connect(this, SIGNAL(width_measure(float, double)), this, SLOT(display_width(float, double)));
    connect(ui.lineEdit_7, SIGNAL(returnPressed()), this, SLOT(show_resize()));
    connect(ui.lineEdit_6, SIGNAL(returnPressed()), this, SLOT(show_angle()));

    connect(ui.action, SIGNAL(triggered()), this, SLOT(show_the_file()));
    connect(ui.action_8, SIGNAL(triggered()), this, SLOT(save_file()));
    connect(ui.action_4, SIGNAL(triggered()), this, SLOT(close_app()));


    //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
    connect(ui.verticalSlider, &QSlider::valueChanged, this, [this](int value) {    
        ui.label_46->setText(QString::number(value-150));
    });
      
    connect(ui.verticalSlider_3, &QSlider::valueChanged, this, [this](int value) {
        
        ui.label_47->setText(QString::number(value-150));
    });
    connect(ui.verticalSlider_2, &QSlider::valueChanged, this, [this](int value) {
        ui.label_48->setText(QString::number(value-150));
    });

    //ï¿½ï¿½ï¿½ï¿½Ä¬ï¿½ï¿½Öµ
    ui.lineEdit->setPlaceholderText("40");
    ui.lineEdit_2->setPlaceholderText("480");
    ui.lineEdit_3->setPlaceholderText("320");
    ui.lineEdit_5->setPlaceholderText("63");
    ui.lineEdit_6->setPlaceholderText("0");
    ui.lineEdit_7->setPlaceholderText("100");

    //ï¿½ï¿½ï¿½ï¿½×´Ì¬ï¿½Æ³ï¿½Ì¬
    setLED(ui.label_9, 1, 16);
    setLED(ui.label_40, 2, 16);
}

Qimage_process::~Qimage_process()
{

    
}

void Qimage_process::timer_Update()
{
    QDateTime time = QDateTime::currentDateTime();
    //ï¿½ï¿½-ï¿½ï¿½-ï¿½ï¿½ Ê±ï¿½ï¿½ï¿½Ö£ï¿½ï¿½ï¿½ ï¿½ï¿½ï¿½ï¿½
    QString str = time.toString("yyyy-MM-dd hh:mm:ss dddd");
    ui.label_43->setText(str);
}


void Qimage_process::display_speed(double times1, double times2, double times3, double times4, double times5, double times6, float cost_time, float gtimes1, float gtimes2, float gtimes3, float gtimes4, float gtimes5) {
    
    //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Í¼

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
    chart->legend()->setVisible(true); //ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½Îªï¿½ï¿½Ê¾×´Ì¬
    chart->legend()->setAlignment(Qt::AlignLeft);//ï¿½ï¿½ï¿½ï¿½Í¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ê¾Î»ï¿½ï¿½ï¿½Úµ×²ï¿½
    chart->legend()->setContentsMargins(5, 0, 0, 0);  

    QChartView* chartView = new QChartView();
    ui.graphicsView->setChart(chart);

    QValueAxis* axisX = new QValueAxis;  //Xï¿½ï¿½
    axisX->setRange(0, 50);   //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½á·¶Î§
    axisX->setTitleText("time(ms)");  //ï¿½ï¿½ï¿½ï¿½
    axisX->setLabelFormat("%.2f");   //ï¿½ï¿½Ç©ï¿½ï¿½Ê½
    axisX->setTickCount(5);    //ï¿½ï¿½ï¿½Ö¸ï¿½ï¿½ï¿½ï¿½ï¿½
    axisX->setMinorTickCount(1);    //Ã¿ï¿½ï¿½ï¿½ï¿½Î»Ö®ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ë¶ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
    chart->setAxisX(axisX, series1);
    chart->setAxisX(axisX, series2);

    ui.graphicsView->setRenderHint(QPainter::Antialiasing);
    ui.graphicsView->setRubberBand(QChartView::NoRubberBand);

    //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½×´Í¼
    QBarSet* set0 = new QBarSet("cpu");
    QBarSet* set1 = new QBarSet("cuda");

    *set0 << times1 << times2 << times3 << times4 << times5 << times6;
    *set1 << gtimes1 << gtimes2 << gtimes3 << gtimes4 << gtimes5 << 0;

    QBarSeries* series = new QBarSeries();
    series->append(set0);
    series->append(set1);

    QChart* chartc = new QChart();
    ui.graphicsView_2->setChart(chartc);

    QValueAxis* axisY = new QValueAxis;  //Yï¿½ï¿½
    axisY->setRange(0, 18);
    axisY->setTitleText("time(ms)");
    axisY->setLabelFormat("%.1f"); 
    axisY->setTickCount(5); 
    axisY->setMinorTickCount(1); 
    chartc->setAxisY(axisY, series);

    chartc->addSeries(series);

    chartc->setTitle(QString::fromLocal8Bit("cpuÓëcudaÍ¼Ïñ´¦ÀíÊ±¼ä¶Ô±È"));
    chartc->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << QString::fromLocal8Bit("»Ò¶È»¯")
               << QString::fromLocal8Bit("ãÐÖµ»¯")
               << QString::fromLocal8Bit("ÂË²¨")
               << QString::fromLocal8Bit("ÐÎÌ¬Ñ§")
               << QString::fromLocal8Bit("±ßÔµ¼ì²â")
               << QString::fromLocal8Bit("¾àÀë±ä»»");

    QBarCategoryAxis* axis = new QBarCategoryAxis();
    axis->append(categories);
    QFont font;
    font.setPointSize(6);  // ÉèÖÃ×ÖÌå´óÐ¡
    axis->setLabelsFont(font);
    //chartc->createDefaultAxes();//´´½¨Ä¬ÈÏµÄ×ó²àµÄ×ø±êÖá£¨¸ù¾Ý QBarSet ÉèÖÃµÄÖµ£©
    
    chartc->setAxisX(axis, series);//ÉèÖÃ×ø±êÖá

    chartc->legend()->setVisible(false); //ÉèÖÃÍ¼ÀýÎªÏÔÊ¾×´Ì¬
    chartc->legend()->setAlignment(Qt::AlignRight);//ÉèÖÃÍ¼ÀýµÄÏÔÊ¾Î»ÖÃÔÚµ×²¿

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

    init_pushButton();
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
    image_ori = image_ori.transformed(matrix, Qt::FastTransformation);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    init_pushButton();
}

void Qimage_process::horizenal_flip() {
    image_ori = image_ori.mirrored(true, false);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    init_pushButton();
}

void Qimage_process::vertical_flip() {
    image_ori = image_ori.mirrored(false, true);
    image_ori = image_ori.convertToFormat(QImage::Format_RGB888);
    ui.label_5->setPixmap(QPixmap::fromImage(image_ori));

    init_pushButton();
}

void Qimage_process::saturation(int sat) { 
   
    QColor oldColor;
    QColor newColor;
    int h, s, l;

    for (int x = 0; x < image_ori.width(); x++) {
        for (int y = 0; y < image_ori.height(); y++) {
            oldColor = QColor(image_ori.pixel(x, y));

            newColor = oldColor.toHsl();
            h = newColor.hue();
            s = newColor.saturation() + sat;
            l = newColor.lightness();

            //we check if the new value is between 0 and 255  
            s = qBound(0, s, 255);

            newColor.setHsl(h, s, l);

            image_ori.setPixel(x, y, qRgb(newColor.red(), newColor.green(), newColor.blue()));
            ui.label_5->setPixmap(QPixmap::fromImage(image_ori));
        }
    }
    
}

void Qimage_process::renew(int xnull) {
    
    int bri = 0;
    int cont = 0;
    bri = ui.verticalSlider->value();
    cont = ui.verticalSlider_3->value();
    uint8_t* rgb = image_ori.bits();
    int r = 0, g = 0, b = 0;
    int size = image_ori.width() * image_ori.height();
    for (int i = 0; i < size; i++) {
        r = bri * 0.01 * rgb[i * 3] - 150 + cont;
        g = bri * 0.01 * rgb[i * 3 + 1] - 150 + cont;
        b = bri * 0.01 * rgb[i * 3 + 2] - 150 + cont;
        r = qBound(0, r, 255);
        g = qBound(0, g, 255);
        b = qBound(0, b, 255);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
        ui.label_5->setPixmap(QPixmap::fromImage(image_ori));
    }
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

    init_pushButton();
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

        // ï¿½ï¿½ï¿½Ä¼ï¿½ï¿½ï¿½ï¿½ï¿½Ê¾ï¿½Ú±ï¿½Ç©ï¿½ï¿½
        ui.label_42->setText(baseName);

}

void Qimage_process::new_window() {

    ori_display->show();
    QtConcurrent::run(camera_on,ori_display);
    
    ori_display->setText(ui.label_42->text());  
}

void Qimage_process::mouseMoveEvent(QMouseEvent* event)
{
    //ui.label_44->setText(QString::number(event->x()));//ÒÔ´°¿Ú×óÉÏ½ÇÎª0µã
    if (event->button() == Qt::LeftButton) {
        // »ñÈ¡Êó±êµã»÷µÄÏà¶Ô×ø±ê
        QPoint pos = mapFromGlobal(QCursor::pos());
        int x = pos.x();
        int y = pos.y();
        ui.label_44->setText(QString::number(x));
        //ui.label_44->setText(QString::number(event->x()));

    }

        //// »ñÈ¡Í¼ÏñÏñËØÖµ
        //QImage image = pixmap.toImage(); // pixmapÊÇÒ»¸öQPixmap¶ÔÏó
        //QRgb pixel = image.pixel(x, y);
        //int r = qRed(pixel);
        //int g = qGreen(pixel);
        //int b = qBlue(pixel);
        //qDebug() << "Pixel value at (" << x << "," << y << "): (" << r << "," << g << "," << b << ")";

    

    //    ui->showX->setText(QString::number(QCursor().pos().x()));//ï¿½Ôµï¿½ï¿½ï¿½ï¿½ï¿½Ä»ï¿½ï¿½ï¿½Ï½ï¿½Îª0ï¿½ï¿½
    //    ui->showY->setText(QString::number(QCursor().pos().y()));

    //  ui->showdata->setText(tr("(%1,%2)").arg(event->x()).arg(event->y())); //Å¶¡£¡£Ò»¸ö±êÇ©ÏÔÊ¾Á½¸ö±äÁ¿ÊÇÕâÃ´¸ãµÄ ²»ºÃÒâË¼¶ªÈËÁË
    //arg()ÊÇQStringÀàÖÐµÄÒ»¸ö¾²Ì¬º¯Êý£¬Ê¹ÓÃËü¾Í¿ÉÒÔÔÚ×Ö·û´®ÖÐÊ¹ÓÃ±äÁ¿ÁË¡£ËùÒÔ¾Í²»ÓÃÄÇ¸öÇ¿ÖÆÀàÐÍ×ª»»ÁË
    //QColor pixcolor = QColor(pool.pixel(event->x(), event->y()));//ÓÐpool.pixelColor()²»ÖªµÀ¸ÉÂïµÄ
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
    
    ui.label_3->clear();
    ui.label->clear();
    ui.label_5->clear();
    ui.verticalSlider->setValue(150);
    ui.verticalSlider_2->setValue(150);
    ui.verticalSlider_3->setValue(150);
}

void Qimage_process::close_app() {
    this->close();
}

void Qimage_process::gray_image(bool gray_on) {
    if (gray_on) {
        //CUDA:to gray_image
        unsigned char* d_rgb;
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
    float* filter = new float[filterWidth * filterWidth];   //Éú³É¸ßË¹ºË
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
            filter[i * filterWidth + j] *= sum1;  //¸ßË¹¾í»ýºË¹éÒ»»¯
        }
    }

    if (filter_on) {
        //CUDA:to filter_image
        ui.label_3->clear();

        unsigned char* d_gray;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gray, q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gray, q_image_gray.bits(), q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_gauss;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gauss, q_image_gray.width() * q_image_gray.height() * sizeof(unsigned char)));

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
        //QString filename = "F:/about_lesson/Qimage_process/56.jpg"; // Òªï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä¼ï¿½ï¿½ï¿?
        //filter_cpu_image.save(filename, "jpg");
        
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

        unsigned char* d_gauss;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gauss, q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_gauss, q_image_gauss.bits(), q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_thresh;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_thresh, q_image_gauss.width() * q_image_gauss.height() * sizeof(unsigned char)));

        float* d_sum;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sum, 256 * sizeof(float)));
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
        QString th = ui.lineEdit->text();
        bool ok;                                   
        int thresh_value = th.toInt(&ok, 10);
        if (thresh_value < 0 || thresh_value > 255) {
            return;
        }
        ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
        //QString filename = "F:/about_lesson/Qimage_process/66.jpg"; // Òªï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä¼ï¿½ï¿½ï¿?
        //filter_cpu_image.save(filename, "jpg");
        //cv::Mat filter_mat = cv::Mat(filter_cpu_image.height(), filter_cpu_image.width(), CV_8UC1,  (void*)filter_cpu_image.bits(), filter_cpu_image.bytesPerLine());
        //imshow("", filter_mat);
        //cv::Mat binary_mat(filter_mat.cols, filter_mat.rows, CV_8UC1);
        //imshow("", filter_mat);
        //threshold(filter_mat, binary_mat,thresh_value, 255, THRESH_BINARY /*| THRESH_OTSU*/);
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
        ui.label_3->setPixmap(QPixmap(""));

        unsigned char* d_thresh;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_thresh, q_image_thresh.width() * q_image_thresh.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_thresh, q_image_thresh.bits(), q_image_thresh.width() * q_image_thresh.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_dil;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dil, q_image_thresh.width() * q_image_thresh.height() * sizeof(unsigned char)));
        unsigned char* d_closed;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, q_image_thresh.width() * q_image_thresh.height() * sizeof(unsigned char)));

        CHECK_CUDA_ERROR(dilation(d_thresh, d_dil, q_image_thresh.width(), q_image_thresh.height()));
        CHECK_CUDA_ERROR(erosion(d_dil, d_closed, q_image_thresh.width(), q_image_thresh.height()));

        q_image_closed = QImage(q_image_thresh.width(), q_image_thresh.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_closed.bits(), d_closed, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_closed));
        QCoreApplication::processEvents();

        CHECK_CUDA_ERROR(cudaFree(d_dil));
        CHECK_CUDA_ERROR(cudaFree(d_thresh));
        CHECK_CUDA_ERROR(cudaFree(d_closed));

        //cpu:to closed_image
        /*cv::Mat binary_mat(binary_cpu_image.height(), binary_cpu_image.width(), CV_8UC1, (void*)binary_cpu_image.bits(), binary_cpu_image.bytesPerLine());
        cv::Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
        cv::Mat closed_mat(binary_mat.size(), CV_8UC1, Scalar(0));
        morphologyEx(binary_mat, closed_mat, MORPH_CLOSE, element);
        closed_cpu_image = QImage(closed_mat.data, closed_mat.cols, closed_mat.rows, QImage::Format_Grayscale8);*/

        //ui.label->setPixmap(QPixmap::fromImage(closed_cpu_image));
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_thresh));
        //ui.label->setPixmap(QPixmap::fromImage(binary_cpu_image));
    }
}

void Qimage_process::canny_image(bool canny_on) {
    
    //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Öµ
    int t_high = 150;
    int t_low = 50;
    if (canny_on) {
        //CUDA:to canny_image
        ui.label_3->clear();
        ui.label_3->setPixmap(QPixmap());

        unsigned char* d_closed;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_closed, q_image_closed.bits(), q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_sobel;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sobel, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));
        unsigned char* d_nms;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nms, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));
        unsigned char* d_high;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_high, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));
        unsigned char* d_canny;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_canny, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));

        int* d_gx;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gx, q_image_closed.width() * q_image_closed.height() * sizeof(int)));
        int* d_gy;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gy, q_image_closed.width() * q_image_closed.height() * sizeof(int)));
        unsigned* d_map;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_map, q_image_closed.width() * q_image_closed.height() * sizeof(d_map[0])));

        CHECK_CUDA_ERROR(sobel_intensity_gradient(d_closed, d_sobel, d_gx, d_gy, q_image_closed.width(), q_image_closed.height()));
        CHECK_CUDA_ERROR(non_max(d_sobel, d_nms, d_gx, d_gy, q_image_closed.width(), q_image_closed.height()));
        CHECK_CUDA_ERROR(hysteresis(d_nms, d_high, d_canny, d_map, t_high, t_low, q_image_closed.width(), q_image_closed.height()));

        q_image_canny = QImage(q_image_closed.width(), q_image_closed.height(), QImage::Format_Grayscale8);
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
        cv::Mat filter_mat(gray_mat.cols, gray_mat.rows, CV_8UC1);
        GaussianBlur(gray_mat, filter_mat, Size(5, 5), 0, 0);

        cv::Mat binary_mat(filter_mat.cols, filter_mat.rows, CV_8UC1);
        threshold(filter_mat, binary_mat, 40, 255, THRESH_BINARY /*| THRESH_OTSU*/);

        cv::Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
        cv::Mat closed_mat(binary_mat.size(), CV_8UC1, Scalar(0));
        morphologyEx(binary_mat, closed_mat, MORPH_CLOSE, element);

        cv::Mat canny_mat(closed_mat.cols, closed_mat.rows, CV_8UC1, Scalar(0));
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
    else 
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_closed));
}

void Qimage_process::distancetransform(bool dist_on) {
    if (dist_on) {
        //CUDA:to dist_image
        ui.label_3->clear();
        ui.label_3->setPixmap(QPixmap());

        unsigned char* d_closed;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_closed, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_closed, q_image_closed.bits(), q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned char* d_distrans;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_distrans, q_image_closed.width() * q_image_closed.height() * sizeof(unsigned char)));

        int* d_max_x;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_x, 256 * sizeof(int)));
        int* d_max_y;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_max_y, 256 * sizeof(int)));

        CHECK_CUDA_ERROR(distance_transform(d_closed, d_distrans, q_image_closed.width(), q_image_closed.height(), d_max_x, d_max_y));
         
        q_image_distrans = QImage(q_image_closed.width(), q_image_closed.height(), QImage::Format_Grayscale8);
        CHECK_CUDA_ERROR(cudaMemcpy(q_image_distrans.bits(), d_distrans, q_image_distrans.width() * q_image_distrans.height() * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_distrans));
        QCoreApplication::processEvents();

        CHECK_CUDA_ERROR(cudaFree(d_closed));
        CHECK_CUDA_ERROR(cudaFree(d_distrans));
        CHECK_CUDA_ERROR(cudaFree(d_max_x));
        CHECK_CUDA_ERROR(cudaFree(d_max_y));

        //CPU:to dist_image
    }
    else {
        ui.label_3->setPixmap(QPixmap::fromImage(q_image_closed));
        //ui.label->setPixmap(QPixmap::fromImage(filter_cpu_image));
    }
}