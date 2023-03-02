import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import pandas as pd
import numpy as np

# import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

matplotlib.use('Qt5Agg') # although PyQt6

# PyQt6!
from PyQt6 import QtGui
# from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QApplication,
    QWidget, QMessageBox, QCheckBox,
    QDial,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout)

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import QCoreApplication, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# global Vars
apptitle = 'Power Generation'
powerdata = []
predictedpower = 7.31        # CHANGE THIS VALUE!!!!
# taken from the csv file, may be adjusted after reading CSV
minrad = 0
maxrad = 40

# global functions

# function, that creates a string
# and returns the first 4 chars
"""def first4chars(x):
    x = str(x)
    return x[:4]"""

# from Geron example
# extra code – code to save the figures as high-res PNGs for the book
IMAGES_PATH = Path() / "images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)
def save_fig(fig, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    # added by UG, see https://mldoodles.com/matplotlib-saves-blank-plot/
    # https://pythonguides.com/matplotlib-savefig-blank-image/
    # https://stackabuse.com/save-plot-as-image-with-matplotlib/
    # fig, ax = plt.subplots()
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, format=fig_extension, dpi=resolution) # ax=self.sc.axes, 


# Classes

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, paDC_POWER=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        super(MplCanvas, self).__init__(self.figure)

class SecondWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # use global data
        global powerdata
        
        wid = QWidget()
        # Vertical Layout
        layout2 = QVBoxLayout()
        # Create a new Canvas
        self.sc2 = MplCanvas(self, width=5, height=4, dpi=100)
        # https://stackoverflow.com/questions/32371571/gridspec-of-multiple-subplots-the-figure-containing-the-passed-axes-is-being-cl
        sns.regplot(x = powerdata.IRRADIATION, y = powerdata.DC_POWER, ax=self.sc2.axes)
        save_fig(self.sc2.figure,"scatterplot", tight_layout=True, fig_extension="png", resolution=300)  # extra code
        layout2.addWidget(self.sc2)
        self.setCentralWidget(wid)
        wid.setLayout(layout2)
   

class Window(QMainWindow):

    def __init__(self):
      super().__init__()
      self.loadData()
      self.initUI()

    def loadData(self):
        global powerdata
        
        self.powerdata = pd.read_csv('plant_training_data.csv') # , delimiter=';'
        # take cols 1-7 (without index)
        #X2 = self.powerdata.IRRADIATION
        # X2 must be converted to array
        # https://stackoverflow.com/questions/69326639/sklearn-warning-valid-feature-names-in-version-1-0
        #X2 = X2.values
        #Y2 = self.powerdata[["DC_POWER"]]
        #Y2 = Y2.values
        print("self.powerdata.head(8):")
        print(self.powerdata.head(8))
        print("self.powerdata.info():")
        self.powerdata.info()
        print("self.powerdata.describe():")
        print(self.powerdata.describe())
        print("powerdata.corr():")
        corr_matrix = self.powerdata.corr()
        print(corr_matrix)
        
        # assign to global variable
        powerdata = self.powerdata
        
        # Split the targets into training/testing sets
        # diabetes_y_train = diabetes_y[:-20]
        # diabetes_y_test = diabetes_y[-20:]
        x = powerdata.IRRADIATION
        y = powerdata.DC_POWER
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
        arr = x_train.array
        x_train = arr.reshape(-1,1)
        arr = x_test.array
        x_test = arr.reshape(-1, 1)
        # Create linear regression object
        self.regr = linear_model.LinearRegression()

        # Train the model using the training sets
        self.regr.fit(x_train, y_train)

        # Make predictions using the testing set
        Y_pred = self.regr.predict(x_test)
        self.test_y = y_test

        # The coefficients
        print("Coefficients: \n", self.regr.coef_) 
        # The mean squared error
        # print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
        print("Mean squared error: %.2f" % mean_squared_error(self.test_y, Y_pred))
        # The coefficient of determination: 1 is perfect prediction
        # print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
        print("Coefficient of determination: %.2f" % r2_score(self.test_y, Y_pred))

    def show_second_window(self):
        if self.w2.isHidden(): 
            self.w2.show()
    
    def initUI(self):
        global minrad
        global maxrad
        global predictedpower

        print("initUI")
      #try:
        self.w2 = SecondWindow()
        # set Appilcation default styles
        font = QtGui.QFont("Sanserif", 12)
        # font.setStyleHint(QtGui.QFont.)
        QApplication.setFont(QtGui.QFont(font))
        QApplication.setWindowIcon(QtGui.QIcon('application-document.png'))

        # grid layout
        layout = QGridLayout()

        # 1st row of GridLayout
        # Dialer
        self.qd = QDial()
        self.qd.setMinimum(0)
        self.qd.setMaximum(46)
        self.qd.setValue(30)
        self.qd.valueChanged.connect(self.updateRadiationAmount) # self = MainWindow
        layout.addWidget(self.qd, 0, 0)
        # Slider
        """
        self.tsl = QSlider(Qt.Orientation.Horizontal)
        self.tsl.setMinimum(minrad) # min bjahr
        self.tsl.setMaximum(maxrad) # max bjahr
        self.tsl.valueChanged.connect(self.updateSelectedIRRADIATION)
        layout.addWidget(self.tsl, 0,1) """

        # 2nd row of GridLayout
        self.radAmount = QLabel(self)
        self.radAmount.setText(" IRRADIATION Amount : " + str(self.qd.value()))
        layout.addWidget(self.radAmount, 1,0)
        # in 2nd cell of 2nd row we use a Horizontal Layout for the 2 labels
        """
        layout2 = QHBoxLayout()
        widget2 = QWidget()
        self.minrad = QLabel(self)
        self.maxrad = QLabel(self)
        self.selectedIRRADIATION = QLabel(self)
        self.selectedIRRADIATION.setStyleSheet("QLabel {color: red}")
        self.minrad.setText("Building IRRADIATION from: " + str(minrad) + " (select with Slider)")
        self.maxrad.setText("           to: " + str(maxrad))
        self.selectedIRRADIATION.setText(str(minrad))
        layout2.addWidget(self.minrad)
        layout2.addWidget(self.selectedIRRADIATION)
        layout2.addWidget(self.maxrad)
        widget2.setLayout(layout2)
        layout.addWidget(widget2, 1, 1) """

        # 3rd row of GridLayout
        # self.lab2.setFont(QtGui.QFont("Sanserif", 15))
        # Stack layout for labels in Grid cell 1,0

        layout3 = QVBoxLayout()

        widget3 = QWidget()


        # Slider for position of an appartment
        """
        self.labPosDescr = QLabel(self)
        self.labPosDescr.setText("Location of the Appartment (1-3)")
        layout3.addWidget(self.labPosDescr)
        self.slPos = QSlider(Qt.Orientation.Horizontal)
        self.slPos.setMinimum(1) # min rating
        self.slPos.setMaximum(3) # max rating
        self.slPos.valueChanged.connect(self.updateLabPos)
        layout3.addWidget(self.slPos)
        self.labPos = QLabel(self)
        self.labPos.setText("1")
        layout3.addWidget(self.labPos)
        # Checkboxes
        # Kitchen
        self.cbKitchen = QCheckBox()
        self.cbKitchen.setText("Kitchen")
        layout3.addWidget(self.cbKitchen)
        # Bath
        self.cbBath = QCheckBox()
        self.cbBath.setText("Bath")
        layout3.addWidget(self.cbBath)
        # Central Heating
        self.cbHeat = QCheckBox()
        self.cbHeat.setText("Central Heating")
        layout3.addWidget(self.cbHeat)
        """

        btn = QPushButton("Predict", self)
        btn.setToolTip("Show Prediction")
        # btn.clicked.connect(QCoreApplication.instance().quit)
        # instead of above button signal, try to call closeEvent method below
        btn.clicked.connect(self.showPrediction)

        btn.resize(btn.sizeHint())
        # btn.move(410, 118)

        # close button set underneath the grid
        layout3.addWidget(btn)

        self.powPrediction = QLabel(self)
        self.powPrediction.setText("Predicted Power: " + str(predictedpower))
        layout3.addWidget(self.powPrediction)

        widget3.setLayout(layout3)
        layout.addWidget(widget3, 2, 0)

        # Canvas for plot
        # https://stackoverflow.com/questions/67590023/embedding-dataframe-plot-on-figure-canvas
        # https://matplotlib.org/3.2.2/gallery/misc/plotfile_demo_sgskip.html
        # Create the maptlotlib FigureCanvas object,
        # which defines a single set of axes as self.axes.
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_DC_POWER(0,predictedpower)

        self.sc.setMinimumWidth(100)
        layout.addWidget(self.sc, 2,1)

        widget = QWidget()
        widget.setLayout(layout)

        # Menu
        # First action - Show Historgrams Window
        """
        button_action1 = QAction(QIcon("application-block.png"), "&Histogram", self)
        button_action1.setStatusTip("Show histograms of data")
        button_action1.triggered.connect(self.show_second_window)
        """
        # Second action
        button_action2 = QAction(QIcon("store.png"), "&Save Prediction Image", self)
        button_action2.setStatusTip("Save Image")
        button_action2.triggered.connect(self.sClick)
        button_action2.setCheckable(True)
        # Third action
        button_action3 = QAction(QIcon("external.png"), "&Close", self)
        button_action3.setStatusTip("Close Application")
        button_action3.triggered.connect(self.closeEvent)
        button_action3.setCheckable(True)
        # Menubar
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        # file_menu.addAction(button_action1)
        file_menu.addAction(button_action2)
        file_menu.addAction(button_action3)

        # Set the central widget of the Window. Widget will expand
        # to take up all the space in the window by default.
        self.setCentralWidget(widget)
        self.setWindowTitle(apptitle) # apptitle global var
        self.setGeometry(30, 30, 700, 550)
        self.show()
        return True
        #except:
        #print("Error while initializing the User Interface")
        #sys.exit()

    # function to print DC_POWER against IRRADIATION
    # when IRRADIATION and DC_POWER are set, a marker is set in the plot
    def plot_DC_POWER(self, IRRADIATION=None, DC_POWER=None):
        global powerdata
        # clear canvas
        self.sc.axes.cla()
        # attributes = ["mieteqm", "bjahr"] # , "flaeche", "bjahr"
        # self.df = powerdata[attributes]
        # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy     
        self.df = powerdata.loc[:, ('DC_POWER', 'IRRADIATION')]
        # plot the pandas DataFrame, passing in the
        # matplotlib Canvas axes.
        # https://stackoverflow.com/questions/37787698/how-to-sort-pandas-dataframe-from-one-column
        # SettingWithCopyWarning:
        # A value is trying to be set on a copy of a slice from a DataFrame
        #      c:\Users\ugarmann\Documents\python-projekte\sas-ws-22-23-1\main.py:261: SettingWithCopyWarning: 
        # A value is trying to be set on a copy of a slice from a DataFrame
        
        self.df.sort_values(by=['IRRADIATION'], inplace=True)
        self.df.reset_index(drop = True, inplace = True)
        # self.df.sort_index(inplace=True)
        print("Min IRRADIATION: %s" % self.df['IRRADIATION'].min())
        print("Max IRRADIATION: %s" % self.df['IRRADIATION'].max())
        print(self.df.head(8))
        # https://pandas.pydata.org/docs/getting_started/intro_tutorials/04_plotting.html
        # https://www.tutorialspoint.com/python_pandas/python_pandas_visualization.htm
        self.df.set_index('IRRADIATION').plot(ax=self.sc.axes) # , columns=self.powerdata["bjahr"]
        
        self.sc.axes.plot(IRRADIATION, DC_POWER, marker="*", markersize=5, c='red')
        save_fig(self.sc.figure, "prediction_plot", tight_layout=True, fig_extension="png", resolution=300)  # extra code
        
        self.sc.draw()

    def updateSelectedIRRADIATION(self):
        val = self.tsl.value()
        self.selectedIRRADIATION.setText(str(val))
        print(val)

    def updateRadiationAmount(self):
        val = self.qd.value()
        self.radAmount.setText(" Radiation Amount : " + str(val))
        print(val)

        # self.sc.axes.cla() # not clear()
        # self.sc.axes.plot([0,1,2,3,4], [val,1,20,3,val]) # r? is color
        # self.sc.draw()
    """
    def updateLabPos(self):
        val = self.slPos.value()
        self.labPos.setText(str(val))
        print(val) """
        
    def sClick(self, event):
        save_fig(self.sc.figure, "prediction_plot")
        
    def closeEvent(self, event):
        """Generate 'question' dialog on clicking 'X' button in title bar.

        Reimplement the closeEvent() event handler to include a 'Question'
        dialog with options on how to proceed - Save, Close, Cancel buttons
        """
        reply = QMessageBox.question(
            self, "Message",
            "Are you sure you want to quit? Any unsaved work will be lost.",
            # QMessageBox.StandardButton since PyQt6
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)

        if (reply == QMessageBox.StandardButton.Close)  : 
            print("Close Event reply close")
            sys.exit()        
        else:
            if (reply == QMessageBox.StandardButton.Save): 
                save_fig(self.sc.figure, "prediction_plot")
                sys.exit()
            else:
                print("Cancel Closing")
                if not type(event) == bool:
                    event.ignore()
        
    def showPrediction(self) :
        global predictedpower
        
        s = self.qd.value()
        # y = self.tsl.value()
        # p = self.slPos.value()
        """
        k = 0
        if self.cbKitchen.isChecked:
            k = 1
        b = 0
        if self.cbBath.isChecked:
            b = 1
        h = 0
        if self.cbHeat.isChecked:
            h = 1
            """

        x_test = [s] # , y, p, k, b, h
        arr2 = np.array(x_test)
        x_test = arr2.reshape(-1, 1)
        predictedpower = round(self.regr.predict(x_test)[0], 2)
        print(type(predictedpower))
        print("Predicted DC_POWER: %.2f" % predictedpower)
        self.powPrediction.setText("Predicted DC_POWER: " + str(predictedpower))
        self.plot_DC_POWER(IRRADIATION=s, DC_POWER=predictedpower)

# main
if __name__ == '__main__':
  app = QApplication(sys.argv)
  w = Window() # show is called in initUI()
  sys.exit(app.exec()) # exec_ only with PyQt5