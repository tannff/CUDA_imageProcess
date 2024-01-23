/********************************************************************************
** Form generated from reading UI file 'ori_image.ui'
**
** Created by: Qt User Interface Compiler version 5.15.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ORI_IMAGE_H
#define UI_ORI_IMAGE_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ori_imageClass
{
public:
    QLabel *label;
    QWidget *horizontalLayoutWidget_4;
    QHBoxLayout *horizontalLayout_5;
    QLabel *label_7;
    QComboBox *comboBox;
    QLabel *label_10;
    QLabel *label_9;
    QPushButton *pushButton;
    QLabel *label_2;
    QCheckBox *checkBox;
    QComboBox *comboBox_2;
    QLabel *label_11;
    QLabel *label_12;
    QLabel *label_13;
    QLabel *label_14;
    QLabel *label_15;
    QLabel *label_16;
    QLabel *label_6;
    QLabel *label_8;
    QLabel *label_3;
    QLabel *label_4;
    QSlider *horizontalSlider;
    QLabel *label_5;

    void setupUi(QWidget *ori_imageClass)
    {
        if (ori_imageClass->objectName().isEmpty())
            ori_imageClass->setObjectName(QString::fromUtf8("ori_imageClass"));
        ori_imageClass->resize(883, 509);
        label = new QLabel(ori_imageClass);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(50, 0, 481, 401));
        horizontalLayoutWidget_4 = new QWidget(ori_imageClass);
        horizontalLayoutWidget_4->setObjectName(QString::fromUtf8("horizontalLayoutWidget_4"));
        horizontalLayoutWidget_4->setGeometry(QRect(560, 60, 261, 42));
        horizontalLayout_5 = new QHBoxLayout(horizontalLayoutWidget_4);
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        label_7 = new QLabel(horizontalLayoutWidget_4);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_7->sizePolicy().hasHeightForWidth());
        label_7->setSizePolicy(sizePolicy);
        label_7->setMaximumSize(QSize(80, 40));
        label_7->setFrameShape(QFrame::Panel);
        label_7->setFrameShadow(QFrame::Raised);
        label_7->setAlignment(Qt::AlignCenter);

        horizontalLayout_5->addWidget(label_7);

        comboBox = new QComboBox(horizontalLayoutWidget_4);
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->setObjectName(QString::fromUtf8("comboBox"));
        comboBox->setMinimumSize(QSize(0, 40));

        horizontalLayout_5->addWidget(comboBox);

        label_10 = new QLabel(ori_imageClass);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setGeometry(QRect(680, 180, 91, 40));
        sizePolicy.setHeightForWidth(label_10->sizePolicy().hasHeightForWidth());
        label_10->setSizePolicy(sizePolicy);
        label_10->setMaximumSize(QSize(100, 40));
        label_10->setFrameShape(QFrame::Panel);
        label_10->setFrameShadow(QFrame::Sunken);
        label_10->setAlignment(Qt::AlignCenter);
        label_9 = new QLabel(ori_imageClass);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setGeometry(QRect(560, 170, 80, 39));
        sizePolicy.setHeightForWidth(label_9->sizePolicy().hasHeightForWidth());
        label_9->setSizePolicy(sizePolicy);
        label_9->setMaximumSize(QSize(80, 40));
        label_9->setFrameShape(QFrame::Panel);
        label_9->setFrameShadow(QFrame::Raised);
        label_9->setAlignment(Qt::AlignCenter);
        pushButton = new QPushButton(ori_imageClass);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setGeometry(QRect(560, 400, 261, 31));
        label_2 = new QLabel(ori_imageClass);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setGeometry(QRect(540, 20, 311, 441));
        label_2->setFrameShape(QFrame::Box);
        label_2->setFrameShadow(QFrame::Sunken);
        label_2->setLineWidth(3);
        checkBox = new QCheckBox(ori_imageClass);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));
        checkBox->setGeometry(QRect(560, 30, 71, 16));
        checkBox->setChecked(true);
        comboBox_2 = new QComboBox(ori_imageClass);
        comboBox_2->addItem(QString());
        comboBox_2->addItem(QString());
        comboBox_2->addItem(QString());
        comboBox_2->addItem(QString());
        comboBox_2->setObjectName(QString::fromUtf8("comboBox_2"));
        comboBox_2->setGeometry(QRect(560, 120, 261, 31));
        comboBox_2->setMinimumSize(QSize(0, 20));
        label_11 = new QLabel(ori_imageClass);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setGeometry(QRect(700, 330, 51, 39));
        label_11->setMaximumSize(QSize(16777, 16777215));
        label_11->setAlignment(Qt::AlignCenter);
        label_12 = new QLabel(ori_imageClass);
        label_12->setObjectName(QString::fromUtf8("label_12"));
        label_12->setGeometry(QRect(50, 400, 491, 51));
        label_12->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);
        label_13 = new QLabel(ori_imageClass);
        label_13->setObjectName(QString::fromUtf8("label_13"));
        label_13->setGeometry(QRect(560, 220, 80, 39));
        sizePolicy.setHeightForWidth(label_13->sizePolicy().hasHeightForWidth());
        label_13->setSizePolicy(sizePolicy);
        label_13->setMaximumSize(QSize(80, 40));
        label_13->setFrameShape(QFrame::Panel);
        label_13->setFrameShadow(QFrame::Raised);
        label_13->setAlignment(Qt::AlignCenter);
        label_14 = new QLabel(ori_imageClass);
        label_14->setObjectName(QString::fromUtf8("label_14"));
        label_14->setGeometry(QRect(650, 220, 61, 40));
        sizePolicy.setHeightForWidth(label_14->sizePolicy().hasHeightForWidth());
        label_14->setSizePolicy(sizePolicy);
        label_14->setMaximumSize(QSize(100, 40));
        label_14->setFrameShape(QFrame::Panel);
        label_14->setFrameShadow(QFrame::Sunken);
        label_14->setAlignment(Qt::AlignCenter);
        label_15 = new QLabel(ori_imageClass);
        label_15->setObjectName(QString::fromUtf8("label_15"));
        label_15->setGeometry(QRect(720, 220, 31, 41));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_15->setFont(font);
        label_15->setAlignment(Qt::AlignCenter);
        label_16 = new QLabel(ori_imageClass);
        label_16->setObjectName(QString::fromUtf8("label_16"));
        label_16->setGeometry(QRect(760, 220, 61, 40));
        sizePolicy.setHeightForWidth(label_16->sizePolicy().hasHeightForWidth());
        label_16->setSizePolicy(sizePolicy);
        label_16->setMaximumSize(QSize(100, 40));
        label_16->setFrameShape(QFrame::Panel);
        label_16->setFrameShadow(QFrame::Sunken);
        label_16->setAlignment(Qt::AlignCenter);
        label_6 = new QLabel(ori_imageClass);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(650, 270, 137, 39));
        label_6->setFrameShape(QFrame::Panel);
        label_6->setFrameShadow(QFrame::Sunken);
        label_8 = new QLabel(ori_imageClass);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(790, 270, 30, 39));
        label_8->setMinimumSize(QSize(10, 0));
        label_8->setMaximumSize(QSize(30, 16777215));
        label_3 = new QLabel(ori_imageClass);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setGeometry(QRect(560, 270, 80, 39));
        label_3->setMinimumSize(QSize(60, 0));
        label_3->setMaximumSize(QSize(80, 16777215));
        label_3->setFrameShape(QFrame::Panel);
        label_3->setFrameShadow(QFrame::Raised);
        label_3->setAlignment(Qt::AlignCenter);
        label_4 = new QLabel(ori_imageClass);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setGeometry(QRect(560, 320, 80, 39));
        label_4->setMinimumSize(QSize(80, 0));
        label_4->setFrameShape(QFrame::Panel);
        label_4->setFrameShadow(QFrame::Raised);
        label_4->setAlignment(Qt::AlignCenter);
        horizontalSlider = new QSlider(ori_imageClass);
        horizontalSlider->setObjectName(QString::fromUtf8("horizontalSlider"));
        horizontalSlider->setGeometry(QRect(650, 330, 135, 22));
        horizontalSlider->setMaximumSize(QSize(135, 16777215));
        horizontalSlider->setMaximum(500000);
        horizontalSlider->setSingleStep(1000);
        horizontalSlider->setPageStep(1000);
        horizontalSlider->setOrientation(Qt::Horizontal);
        label_5 = new QLabel(ori_imageClass);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(790, 320, 20, 39));
        label_5->setMinimumSize(QSize(20, 0));
        label_2->raise();
        label->raise();
        horizontalLayoutWidget_4->raise();
        label_10->raise();
        label_9->raise();
        pushButton->raise();
        checkBox->raise();
        comboBox_2->raise();
        label_11->raise();
        label_12->raise();
        label_13->raise();
        label_14->raise();
        label_15->raise();
        label_16->raise();
        label_6->raise();
        label_8->raise();
        label_3->raise();
        label_4->raise();
        horizontalSlider->raise();
        label_5->raise();

        retranslateUi(ori_imageClass);
        QObject::connect(pushButton, SIGNAL(clicked()), ori_imageClass, SLOT(closed_view()));
        QObject::connect(checkBox, SIGNAL(stateChanged(int)), ori_imageClass, SLOT(camera_status(int)));

        QMetaObject::connectSlotsByName(ori_imageClass);
    } // setupUi

    void retranslateUi(QWidget *ori_imageClass)
    {
        ori_imageClass->setWindowTitle(QCoreApplication::translate("ori_imageClass", "ori_image", nullptr));
        label->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_7->setText(QCoreApplication::translate("ori_imageClass", "\350\277\236\346\216\245\346\226\271\345\274\217", nullptr));
        comboBox->setItemText(0, QCoreApplication::translate("ori_imageClass", "USB\350\277\236\346\216\245", nullptr));
        comboBox->setItemText(1, QCoreApplication::translate("ori_imageClass", "\345\215\203\345\205\206\347\275\221\345\217\243\350\277\236\346\216\245", nullptr));

        label_10->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_9->setText(QCoreApplication::translate("ori_imageClass", "\347\233\270\346\234\272\347\212\266\346\200\201", nullptr));
        pushButton->setText(QCoreApplication::translate("ori_imageClass", "\344\277\235\345\255\230\350\256\276\347\275\256\345\271\266\350\277\224\345\233\236\344\270\273\347\225\214\351\235\242", nullptr));
        label_2->setText(QString());
        checkBox->setText(QCoreApplication::translate("ori_imageClass", "\346\211\223\345\274\200\347\233\270\346\234\272", nullptr));
        comboBox_2->setItemText(0, QCoreApplication::translate("ori_imageClass", "\346\211\253\346\217\217\345\217\257\347\224\250\350\256\276\345\244\207", nullptr));
        comboBox_2->setItemText(1, QCoreApplication::translate("ori_imageClass", "Daheng \346\260\264\346\230\237 ", nullptr));
        comboBox_2->setItemText(2, QCoreApplication::translate("ori_imageClass", "\346\227\240\345\217\257\347\224\250\350\256\276\345\244\207", nullptr));
        comboBox_2->setItemText(3, QString());

        label_11->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_12->setText(QCoreApplication::translate("ori_imageClass", "\346\227\245\345\277\227", nullptr));
        label_13->setText(QCoreApplication::translate("ori_imageClass", "\345\233\276\345\203\217\345\260\272\345\257\270", nullptr));
        label_14->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_15->setText(QCoreApplication::translate("ori_imageClass", "\303\227", nullptr));
        label_16->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_6->setText(QCoreApplication::translate("ori_imageClass", "TextLabel", nullptr));
        label_8->setText(QCoreApplication::translate("ori_imageClass", "\345\270\247/s", nullptr));
        label_3->setText(QCoreApplication::translate("ori_imageClass", "\351\207\207\351\233\206\345\270\247\347\216\207", nullptr));
        label_4->setText(QCoreApplication::translate("ori_imageClass", "\346\233\235\345\205\211\346\227\266\351\227\264", nullptr));
        label_5->setText(QCoreApplication::translate("ori_imageClass", "us", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ori_imageClass: public Ui_ori_imageClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ORI_IMAGE_H
