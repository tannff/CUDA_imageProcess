#include "Qimage_process.h"

QT_CHARTS_USE_NAMESPACE      
using namespace std;
using namespace cv;

// �ú�����label�ؼ����һ��Բ��ָʾ�ƣ���Ҫָ����ɫcolor�Լ�ֱ��size
// color 0:grey 1:red 2:green 3:yellow
// size  ��λ������
void Qimage_process::setLED(QLabel* label, int color, int size)
{
    // ��label�е��������
    label->setText("");
    // �����þ��δ�С
    // ���ui�������õ�label��С����С��Ⱥ͸߶�С�����ν�������Ϊ��С��Ⱥ���С�߶ȣ�
    // ���ui�������õ�label��С����С��Ⱥ͸߶ȴ󣬾��ν�������Ϊ����Ⱥ����߶ȣ�
    QString min_width = QString("min-width: %1px;").arg(size);              // ��С��ȣ�size
    QString min_height = QString("min-height: %1px;").arg(size);            // ��С�߶ȣ�size
    QString max_width = QString("max-width: %1px;").arg(size);              // ��С��ȣ�size
    QString max_height = QString("max-height: %1px;").arg(size);            // ��С�߶ȣ�size
    // �����ñ߽���״���߿�
    QString border_radius = QString("border-radius: %1px;").arg(size / 2);    // �߿���Բ�ǣ��뾶Ϊsize/2
    QString border = QString("border:1px solid black;");                    // �߿�Ϊ1px��ɫ
    // ������ñ�����ɫ
    QString background = "background-color:";
    switch (color) {
    case 0:
        // ��ɫ
        background += "rgb(190,190,190)";
        break;
    case 1:
        // ��ɫ
        background += "rgb(255,0,0)";
        break;
    case 2:
        // ��ɫ
        background += "rgb(0,255,0)";
        break;
    case 3:
        // ��ɫ
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

    //ʵʱʱ����ʾ
    timer_calendar = new QTimer(this);//new������
    connect(timer_calendar, SIGNAL(timeout()), this, SLOT(timer_Update()));//timeout��ʱ�¼�
    timer_calendar->start(1000);//ÿ��һ�����һ�βۺ���
    

    ui.label_6->setText(QString::fromLocal8Bit("ԭʼͼ��"));

    QImage image_cpu_out(QString::fromLocal8Bit("F:\\about_lesson\\Qimage_process\\CUDA_imageProcess\\Qimage_process\\gray_1_cpu.jpg"));
    image_cpu_out = image_cpu_out.scaled(ui.label->size(), Qt::KeepAspectRatio);
    QPixmap pix_image_cpushow = QPixmap::fromImage(image_cpu_out);
    ui.label->setPixmap(pix_image_cpushow);
    ui.label_2->setText(QString::fromLocal8Bit("����CPU����ͼ��ʶ��"));


    QImage image_gpu_out(QString::fromLocal8Bit("F:\\about_lesson\\Qimage_process\\CUDA_imageProcess\\Qimage_process\\gray_1_cuda.jpg"));
    image_gpu_out = image_gpu_out.scaled(ui.label_3->size(), Qt::KeepAspectRatio);
    QPixmap pix_image_gpushow = QPixmap::fromImage(image_gpu_out);
    ui.label_3->setPixmap(pix_image_gpushow);
    ui.label_4->setText(QString::fromLocal8Bit("����CUDA����ͼ��ʶ��"));
       
    ui.label_15->setText(QString::fromLocal8Bit("1���ף�mm��="));
    ui.label_28->setText(QString::fromLocal8Bit("���أ�pixel)"));

    QGridLayout* baseLayout = new QGridLayout();//���ֹ�����

    connect(this, SIGNAL(image_proccess_speed(double, double, double, double, double, double, float, float, float, float, float, float)),
            this, SLOT(display_speed(double, double, double, double, double, double, float, float, float, float, float, float)));
    connect(this, SIGNAL(width_measure(float, double)), this, SLOT(display_width(float, double)));
    connect(ui.lineEdit_7, SIGNAL(returnPressed()), this, SLOT(show_resize()));
    connect(ui.lineEdit_6, SIGNAL(returnPressed()), this, SLOT(show_angle()));

    ui.lineEdit->setPlaceholderText("40");
    ui.lineEdit_2->setPlaceholderText("480");
    ui.lineEdit_3->setPlaceholderText("320");
    ui.lineEdit_5->setPlaceholderText("63");
    ui.lineEdit_6->setPlaceholderText("0");
    ui.lineEdit_7->setPlaceholderText("100");

    setLED(ui.label_9, 1, 16);
    setLED(ui.label_40, 2, 16);
    
    QPalette palette;
    palette.setColor(QPalette::Highlight, QColor(255, 255, 0));  // ���û�����ɫ

    ui.verticalSlider->setPalette(palette);  // Ӧ����ʽ
    ui.verticalSlider->setStyleSheet("border-radius: 0px; margin-top: 8px; margin-bottom: 9px;");  // ����������ʽ����
}

Qimage_process::~Qimage_process()
{
    
}

void Qimage_process::timer_Update()
{
    QDateTime time = QDateTime::currentDateTime();
    //��-��-�� ʱ���֣��� ����
    QString str = time.toString("yyyy-MM-dd hh:mm:ss dddd");
    ui.label_43->setText(str);
}


void Qimage_process::display_speed(double times1, double times2, double times3, double times4, double times5, double times6, float cost_time, float gtimes1, float gtimes2, float gtimes3, float gtimes4, float gtimes5) {
    
    //��������ͼ

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
    chart->legend()->setVisible(true); //����ͼ��Ϊ��ʾ״̬
    chart->legend()->setAlignment(Qt::AlignLeft);//����ͼ������ʾλ���ڵײ�
    chart->legend()->setContentsMargins(5, 0, 0, 0);  

    QChartView* chartView = new QChartView();
    ui.graphicsView->setChart(chart);

    QValueAxis* axisX = new QValueAxis;  //X��
    axisX->setRange(0, 50);   //���������᷶Χ
    axisX->setTitleText("time(ms)");  //����
    axisX->setLabelFormat("%.2f");   //��ǩ��ʽ
    axisX->setTickCount(5);    //���ָ�����
    axisX->setMinorTickCount(1);    //ÿ����λ֮������˶���������
    chart->setAxisX(axisX, series1);
    chart->setAxisX(axisX, series2);

    ui.graphicsView->setRenderHint(QPainter::Antialiasing);
    ui.graphicsView->setRubberBand(QChartView::NoRubberBand);

    //������״ͼ
    QBarSet* set0 = new QBarSet("cpu");
    QBarSet* set1 = new QBarSet("cuda");

    *set0 << times1 << times2 << times3 << times4 << times5 << times6;
    *set1 << gtimes1 << gtimes2 << gtimes3 << gtimes4 << gtimes5 << 0;

    QBarSeries* series = new QBarSeries();
    series->append(set0);
    series->append(set1);

    QChart* chartc = new QChart();
    ui.graphicsView_2->setChart(chartc);

    QValueAxis* axisY = new QValueAxis;  //Y��
    axisY->setRange(0, 18);
    axisY->setTitleText("time(ms)");
    axisY->setLabelFormat("%.1f"); 
    axisY->setTickCount(5); 
    axisY->setMinorTickCount(1); 
    chartc->setAxisY(axisY, series);

    chartc->addSeries(series);

    chartc->setTitle(QString::fromLocal8Bit("cpu��cudaͼ����ʱ��Ա�"));
    chartc->setAnimationOptions(QChart::SeriesAnimations);

    QStringList categories;
    categories << QString::fromLocal8Bit("�ҶȻ�")
               << QString::fromLocal8Bit("��ֵ��")
               << QString::fromLocal8Bit("�˲�")
               << QString::fromLocal8Bit("��̬ѧ")
               << QString::fromLocal8Bit("��Ե���")
               << QString::fromLocal8Bit("����任");

    QBarCategoryAxis* axis = new QBarCategoryAxis();
    axis->append(categories);
    QFont font;
    font.setPointSize(6);  // ���������С
    axis->setLabelsFont(font);
    //chartc->createDefaultAxes();//����Ĭ�ϵ����������ᣨ���� QBarSet ���õ�ֵ��
    
    chartc->setAxisX(axis, series);//����������

    chartc->legend()->setVisible(false); //����ͼ��Ϊ��ʾ״̬
    chartc->legend()->setAlignment(Qt::AlignRight);//����ͼ������ʾλ���ڵײ�

    /*QChartView* chartView = new QChartView(chartc);
    chartView = new QChartView();
    baseLayout->addWidget(chartView, 1, 0);
    setLayout(baseLayout);//Qwidget���е�*/

    ui.graphicsView_2->setRenderHint(QPainter::Antialiasing);
    ui.graphicsView_2->setRubberBand(QChartView::NoRubberBand);
     
}

void Qimage_process::show_resize() {

    QString re = ui.lineEdit_7->text();
    bool ok;
    resize_value = re.toInt(&ok,10);
    switch_resize = true;
    emit image_resize(resize_value, switch_resize);
}

void Qimage_process::show_angle() {
    QString an = ui.lineEdit_6->text();
    bool ok;
    float angle_value = an.toFloat(&ok);
    if (angle_value < 0.0 || angle_value > 360.0) {
        return;
    }
    //emit image_resize(angle_value);
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
    emit image_ROI(ROI_width, ROI_height);
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
    QImage image_ori(fileName);
    image_ori = image_ori.scaled(ui.label->size(), Qt::KeepAspectRatio);
    QPixmap pix_image_ori = QPixmap::fromImage(image_ori);
    ui.label_5->setPixmap(pix_image_ori);

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

        // ���ļ�����ʾ�ڱ�ǩ��
        ui.label_42->setText(baseName);

}


