# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PlotUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_ChildrenForm(object):
    def setupUi(self, ChildrenForm):
        ChildrenForm.setObjectName("ChildrenForm")
        ChildrenForm.resize(800, 583)
        self.centralwidget = QtWidgets.QWidget(ChildrenForm)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 60, 741, 471))
        self.graphicsView.setObjectName("graphicsView")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(30, 20, 121, 21))
        self.comboBox.setObjectName("comboBox")
        ChildrenForm.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ChildrenForm)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        ChildrenForm.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ChildrenForm)
        self.statusbar.setObjectName("statusbar")
        ChildrenForm.setStatusBar(self.statusbar)

        self.retranslateUi(ChildrenForm)
        QtCore.QMetaObject.connectSlotsByName(ChildrenForm)

    def retranslateUi(self, ChildrenForm):
        _translate = QtCore.QCoreApplication.translate
        ChildrenForm.setWindowTitle(_translate("ChildrenForm", "MainWindow"))

