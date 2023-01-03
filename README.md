# predecting-dyslexia-using-ML
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyM3r6lvae1uB7NX6ZBNDLPh",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Neeraj0906/LUassignment_sum/blob/main/spt2dys.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "aNimcCxjqj8Z"
      },
      "outputs": [],
      "source": [
        "from sklearn.svm import SVC\n",
        "from sklearn.model_selection import train_test_split\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import f1_score\n",
        "from sklearn.metrics import precision_recall_fscore_support\n",
        "from sklearn.metrics import confusion_matrix, plot_confusion_matrix\n",
        "import matplotlib.pyplot as plt# doctest: +SKIP\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "%matplotlib inline"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Reading the dataset\n",
        "data=pd.read_csv('/content/dysx - Copy.csv')\n",
        "#Value to be predicted by the model.\n",
        "y=data.Label \n",
        "#Input taken by the model.\n",
        "X=data.drop(['Label'],axis=1) \n",
        "data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "XG8oPHk7qyZg",
        "outputId": "12feb0c6-c1b0-46c2-bfd2-a58d4e64976d"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Language_vocab  Memory  Speed  Visual_discrimination  Audio_Discrimination  \\\n",
              "0             0.5     0.6    0.5                    0.8                   0.6   \n",
              "1             0.6     0.7    0.8                    0.9                   0.5   \n",
              "2             0.6     0.4    0.3                    0.3                   0.4   \n",
              "3             0.3     0.5    0.2                    0.1                   0.3   \n",
              "4             0.7     0.6    0.7                    0.8                   0.9   \n",
              "\n",
              "   Survey_Score  Label  \n",
              "0           0.7      1  \n",
              "1           0.8      2  \n",
              "2           0.6      1  \n",
              "3           0.5      0  \n",
              "4           0.5      2  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-8d2f569d-c168-4cb5-b2a4-6f3df1b5197b\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Language_vocab</th>\n",
              "      <th>Memory</th>\n",
              "      <th>Speed</th>\n",
              "      <th>Visual_discrimination</th>\n",
              "      <th>Audio_Discrimination</th>\n",
              "      <th>Survey_Score</th>\n",
              "      <th>Label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.5</td>\n",
              "      <td>0.6</td>\n",
              "      <td>0.5</td>\n",
              "      <td>0.8</td>\n",
              "      <td>0.6</td>\n",
              "      <td>0.7</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.6</td>\n",
              "      <td>0.7</td>\n",
              "      <td>0.8</td>\n",
              "      <td>0.9</td>\n",
              "      <td>0.5</td>\n",
              "      <td>0.8</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.6</td>\n",
              "      <td>0.4</td>\n",
              "      <td>0.3</td>\n",
              "      <td>0.3</td>\n",
              "      <td>0.4</td>\n",
              "      <td>0.6</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.3</td>\n",
              "      <td>0.5</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0.1</td>\n",
              "      <td>0.3</td>\n",
              "      <td>0.5</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.7</td>\n",
              "      <td>0.6</td>\n",
              "      <td>0.7</td>\n",
              "      <td>0.8</td>\n",
              "      <td>0.9</td>\n",
              "      <td>0.5</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-8d2f569d-c168-4cb5-b2a4-6f3df1b5197b')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-8d2f569d-c168-4cb5-b2a4-6f3df1b5197b button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-8d2f569d-c168-4cb5-b2a4-6f3df1b5197b');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#The following test cases will be used to check the values given by each model.\n",
        "test1 = np.array([[0.5, 0.1, 0.2, 0.8, 0.3, 0.5]]) #Readings for applicant 1.\n",
        "test2 = np.array([[0.7, 0.9, 0.4, 0.9, 0.3, 0.8]]) #Readings for applicant 2.\n",
        "test3 = np.array([[0.1, 0.7, 0.2, 0.6, 0.9, 0.6]]) #Readings for applicant 3.\n",
        "test4 = np.array([[0.3, 0.4, 0.5, 0.3, 0.3, 0.5]]) #Readings for applicant 4."
      ],
      "metadata": {
        "id": "DLVWBNMXq1oG"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating the test and train data sets for the given data.\n",
        "X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=10)\n",
        "#list to store precision values of each model.\n",
        "precision = [0, 0, 0, 0, 0]\n",
        "#list to store recall values of each model.\n",
        "recall = [0, 0, 0, 0, 0]\n",
        "#list to store f1-score values of each model.\n",
        "fscore = [0, 0, 0, 0, 0]\n",
        "#list to store error in predictions of each model.\n",
        "error = [.0, .0, .0, .0, .0]"
      ],
      "metadata": {
        "id": "gVLciVxPq6GO"
      },
      "execution_count": 30,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#StandardScalar is used for preprocessing of data.\n",
        "#'copy' is False, which means copies are avoid and inplace scaling is done instead.\n",
        "sc=StandardScaler(copy=False)\n",
        "sc.fit_transform(X_train)\n",
        "sc.transform(X_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UhG4Cmz9q9K3",
        "outputId": "660da73a-68e6-49ae-cbc3-dcf2a9ebc7ab"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.40859278,  0.09423675,  0.57481025,  1.09456548,  0.51247074,\n",
              "         0.54374235],\n",
              "       [ 2.0828266 ,  0.54298319, -0.30276265,  0.14688974,  0.01971041,\n",
              "         0.54374235],\n",
              "       [ 0.0896911 ,  0.09423675,  0.1360238 , -1.27462388, -0.47304992,\n",
              "         0.1016754 ],\n",
              "       ...,\n",
              "       [-0.90687665, -0.80325613, -0.7415491 , -1.74846175, -1.45857058,\n",
              "        -1.22452545],\n",
              "       [-0.90687665,  0.54298319, -0.30276265,  0.14688974, -0.47304992,\n",
              "         0.1016754 ],\n",
              "       [ 0.58797497,  0.09423675,  0.57481025,  2.04224123, -0.47304992,\n",
              "         0.54374235]])"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating lists of label outputs given in each test case by different models\n",
        "label_1 = [0, 0, 0, 0, 0]\n",
        "label_2 = [0, 0, 0, 0, 0]\n",
        "label_3 = [0, 0, 0, 0, 0]\n",
        "label_4 = [0, 0, 0, 0, 0]"
      ],
      "metadata": {
        "id": "hC7ubV0yq_rK"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Decision tree"
      ],
      "metadata": {
        "id": "7e9AFp8rrKzp"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "dt = DecisionTreeClassifier(random_state = 1)\n",
        "#Training the model.\n",
        "dt.fit(X_train, y_train)\n",
        "#Making predictions using the decision tree model.\n",
        "pred_dt = dt.predict(X_test)\n",
        "#Calculating error\n",
        "error[0] = round(mean_absolute_error(y_test, pred_dt), 3)"
      ],
      "metadata": {
        "id": "I543atU4rCKu"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Testing the model with predefined test cases.\n",
        "ans_1 = dt.predict((test1))\n",
        "ans_2 = dt.predict((test2))\n",
        "ans_3 = dt.predict((test3))\n",
        "ans_4 = dt.predict((test4))\n",
        "\n",
        "#Storing the above predictions into respective lists.\n",
        "label_1[0] = ans_1[0]\n",
        "label_2[0] = ans_2[0]\n",
        "label_3[0] = ans_3[0]\n",
        "label_4[0] = ans_4[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xXSa41lFrGVj",
        "outputId": "578e414b-71d9-4918-f46f-5b7a9d9c54d2"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but DecisionTreeClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Random classifier"
      ],
      "metadata": {
        "id": "FlNRNZGCrUJ9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating the RandomClassifier model.\n",
        "rf = RandomForestClassifier(random_state = 0) \n",
        "#Training the model.\n",
        "rf.fit(X_train, y_train)\n",
        "#Making predictions using the model.\n",
        "pred_rf = rf.predict(X_test)\n",
        "#Calculating error\n",
        "error[1] = round(mean_absolute_error(y_test, pred_rf), 3)"
      ],
      "metadata": {
        "id": "gtqCRVs0rSVm"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Testing the model with predefined test cases.\n",
        "ans_1 = rf.predict((test1))\n",
        "ans_2 = rf.predict((test2))\n",
        "ans_3 = rf.predict((test3))\n",
        "ans_4 = rf.predict((test4))\n",
        "\n",
        "#Storing the above predictions into respective lists.\n",
        "label_1[1] = ans_1[0]\n",
        "label_2[1] = ans_2[0]\n",
        "label_3[1] = ans_3[0]\n",
        "label_4[1] = ans_4[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_VifJbf2rdb3",
        "outputId": "b53c7055-608f-4bd4-b585-bc0fd31f2ba0"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "SVM"
      ],
      "metadata": {
        "id": "h3IrM-Sgrh5n"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating the SVM model.\n",
        "svm = SVC(kernel = \"linear\")\n",
        "#Training the model.\n",
        "svm.fit(X_train, y_train)\n",
        "#Making predictions using the model.\n",
        "pred_svm = svm.predict(X_test)\n",
        "#Calculating error\n",
        "error[2] = round(mean_absolute_error(y_test, pred_svm), 3)"
      ],
      "metadata": {
        "id": "kEvIBpwCrgnn"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Testing the model with predefined test cases.\n",
        "ans_1 = svm.predict((test1))\n",
        "ans_2 = svm.predict((test2))\n",
        "ans_3 = svm.predict((test3))\n",
        "ans_4 = svm.predict((test4))\n",
        "\n",
        "#Storing the above predictions into respective lists.\n",
        "label_1[2] = ans_1[0]\n",
        "label_2[2] = ans_2[0]\n",
        "label_3[2] = ans_3[0]\n",
        "label_4[2] = ans_4[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xcIRm91krpvm",
        "outputId": "a39469bd-263d-4a65-b65b-417b53c3473b"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "RandomForest model with GridSearch"
      ],
      "metadata": {
        "id": "GxEyMjXFrw3j"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating a list of possible n_estimators.\n",
        "n_est = {'n_estimators' : [10,100,500,1000]}\n",
        "#Creating a RandomForest model using the value of n_estimators given by GridSearch for best result.\n",
        "rf_grid = GridSearchCV(RandomForestClassifier(random_state=0),n_est,scoring='f1_macro')\n",
        "#Training the model\n",
        "rf_grid.fit(X_train, y_train)\n",
        "#Making predictions using the model.\n",
        "pred_rf_grid = rf_grid.predict(X_test)\n",
        "#Printing the value of n_estimator used in the model.\n",
        "#This value provides the most accurate predictions for our dataset.\n",
        "print('Best value of n_estimator for RandomForest model is:')\n",
        "print(rf_grid.best_params_)\n",
        "#Calculating error\n",
        "error[3] = round(mean_absolute_error(y_test, pred_rf_grid), 3)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WMEQkG22rviD",
        "outputId": "76487c60-cfd7-4301-dc1d-ed9210d25780"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best value of n_estimator for RandomForest model is:\n",
            "{'n_estimators': 100}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "#Testing the model with predefined test cases.\n",
        "ans_1 = rf_grid.predict((test1))\n",
        "ans_2 = rf_grid.predict((test2))\n",
        "ans_3 = rf_grid.predict((test3))\n",
        "ans_4 = rf_grid.predict((test4))\n",
        "\n",
        "#Storing the above predictions into respective lists.\n",
        "label_1[3] = ans_1[0]\n",
        "label_2[3] = ans_2[0]\n",
        "label_3[3] = ans_3[0]\n",
        "label_4[3] = ans_4[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8KzH3NkZr1Tv",
        "outputId": "7f063308-ad1d-44e5-8ef6-689d2cdfb18e"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "SVM model with GridSearch"
      ],
      "metadata": {
        "id": "BfnYVuTYr4-u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#options_parameters is a list of dictionaries to find the most suitable values of 'kernel', 'gamma' and 'C' for the given model.\n",
        "options_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],\n",
        "                     'C': [1, 10, 100, 1000]},\n",
        "                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]\n",
        "#Creating SVM model with the most suitable parameters obtained by using GridSearch.\n",
        "svm_grid = GridSearchCV(SVC(), options_parameters,scoring='f1_macro')\n",
        "#Training the model.\n",
        "svm_grid.fit(X_train, y_train)\n",
        "#Making predictions using the model.\n",
        "pred_svm_grid = svm_grid.predict(X_test)\n",
        "#Printing the values of 'C', 'gamma' and 'kernel' used in our model.\n",
        "#These values provide the most accurate predictions for the given dataset.\n",
        "print('Best parameters of SVM model are:')\n",
        "print(svm_grid.best_params_)\n",
        "#Calculating error\n",
        "error[4] = round(mean_absolute_error(y_test, pred_svm_grid), 3)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "43Pj9c5kr3vj",
        "outputId": "7fb375c0-603e-45b9-9615-c0f416b3b3db"
      },
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Best parameters of SVM model are:\n",
            "{'C': 1, 'kernel': 'linear'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Testing the model with predefined test cases.\n",
        "ans_1 = svm_grid.predict((test1))\n",
        "ans_2 = svm_grid.predict((test2))\n",
        "ans_3 = svm_grid.predict((test3))\n",
        "ans_4 = svm_grid.predict((test4))\n",
        "\n",
        "#Storing the above predictions into respective lists.\n",
        "label_1[4] = ans_1[0]\n",
        "label_2[4] = ans_2[0]\n",
        "label_3[4] = ans_3[0]\n",
        "label_4[4] = ans_4[0]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "knNTiXddr8rL",
        "outputId": "fee1fc20-fe39-48b5-c8ae-590bb4f3fdf5"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but SVC was fitted with feature names\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Comparing Errors of different models"
      ],
      "metadata": {
        "id": "gGtN3b5VsAQg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "models = ['DecisionTree', 'RandomForest','SVM\\t', 'RandomForest\\n(GridSearch)', 'SVM\\n(GridSearch)']\n",
        "print('Model\\t\\tError')\n",
        "for i in range(5):\n",
        "    print('{}\\t{}'.format(models[i],error[i]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TifDqVJkr_aq",
        "outputId": "eb633665-b11a-4a95-b3fc-00ec18a07af2"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model\t\tError\n",
            "DecisionTree\t0.162\n",
            "RandomForest\t0.072\n",
            "SVM\t\t0.075\n",
            "RandomForest\n",
            "(GridSearch)\t0.072\n",
            "SVM\n",
            "(GridSearch)\t0.075\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Comparing Results of Different models"
      ],
      "metadata": {
        "id": "yKoJYndYsHwP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(label_1)\n",
        "print(label_2)\n",
        "print(label_3)\n",
        "print(label_4)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DKFLxMfqsF2v",
        "outputId": "5b6b4e0c-931d-4a6a-aca5-5d88f9ab9b02"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2, 2, 1, 2, 1]\n",
            "[2, 2, 2, 2, 2]\n",
            "[1, 1, 1, 1, 1]\n",
            "[1, 1, 1, 1, 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "plt.figure(figsize=(10,5))\n",
        "sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_1, s = 200, label = 'test1',)\n",
        "sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_2, s = 150, label = 'test2')\n",
        "sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_3, s = 100, label = 'test3')\n",
        "sns.scatterplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_4, s = 50, label = 'test3')\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_1)\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_2)\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_3)\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "                y = label_4)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "id": "HpQ_R_7osMP-",
        "outputId": "7ffe3728-1e7a-467b-df62-bcebd003d5c6"
      },
      "execution_count": 47,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 720x360 with 1 Axes>"
            ],
            "image/png": "=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Confusion Matrices of different models"
      ],
      "metadata": {
        "id": "IW2LjPB9sRnO"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Confusion matrix for Decision tree model.\n",
        "print(confusion_matrix(np.array(y_test), pred_dt))\n",
        "plot_confusion_matrix(dt, X_test, y_test)\n",
        "plt.show()\n",
        "#Finding precision, recall and f-score for Decision Tree Model and updating values in respective lists.\n",
        "precision[0], recall[0], fscore[0], Nil = precision_recall_fscore_support(y_test, pred_dt, average='macro')\n",
        "print('For a DecisionTreeClassifier:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[0], recall[0], fscore[0]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 403
        },
        "id": "jacT0REYsOty",
        "outputId": "4367ac65-66bb-4071-ea67-9ff459f5c1d0"
      },
      "execution_count": 48,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 38  10   0]\n",
            " [  9 187  27]\n",
            " [  0  19 110]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "==\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "For a DecisionTreeClassifier:  Precision = 0.826, Recall = 0.828, F1-score = 0.826\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Confusion matrix for RandomForest model.\n",
        "print(confusion_matrix(np.array(y_test), pred_rf))\n",
        "plot_confusion_matrix(rf, X_test, y_test)\n",
        "plt.show()\n",
        "#Finding precision, recall and f-score for RandomForest Model and updating values in respective lists.\n",
        "precision[1], recall[1], fscore[1], Nil = precision_recall_fscore_support(y_test, pred_rf, average='macro')\n",
        "print('For a RandomForestClassifier:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[1], recall[1], fscore[1]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 404
        },
        "id": "2MYct5aWsUjW",
        "outputId": "0f7532fe-e296-423d-9840-15302ff88184"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 46   2   0]\n",
            " [  3 203  17]\n",
            " [  0   7 122]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "=\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "For a RandomForestClassifier:  Precision = 0.925, Recall = 0.938, F1-score = 0.931\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Confusion matrix for SVM model\n",
        "print(confusion_matrix(np.array(y_test), pred_svm))\n",
        "plot_confusion_matrix(svm, X_test, y_test)\n",
        "plt.show()\n",
        "#Finding precision, recall and f-score for SVM model and updating values in respective lists.\n",
        "precision[2], recall[2], fscore[2], Nil = precision_recall_fscore_support(y_test, pred_svm, average='macro')\n",
        "print('For a SVM model:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[2], recall[2], fscore[2]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 403
        },
        "id": "DXjFQT3ZsXuR",
        "outputId": "a051c721-25d4-479f-a01c-8105a16ccc24"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 44   4   0]\n",
            " [  4 206  13]\n",
            " [  0   9 120]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "For a SVM model:  Precision = 0.920, Recall = 0.924, F1-score = 0.922\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Confusion matrix for RandomForest model using GridSearch\n",
        "print(confusion_matrix(np.array(y_test), pred_rf_grid))\n",
        "plot_confusion_matrix(rf_grid, X_test, y_test)\n",
        "plt.show()\n",
        "#Finding precision, recall and f-score for RandomForest (GridSearch) model and updating values in respective lists.\n",
        "precision[3], recall[3], fscore[3], Nil = precision_recall_fscore_support(y_test, pred_rf_grid, average='macro')\n",
        "print('For a RandomForest model with GridSearch:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[3], recall[3], fscore[3]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 404
        },
        "id": "EeG0xz0isarT",
        "outputId": "6f7c4780-7a9f-4d0a-d510-a19b453fd62e"
      },
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 46   2   0]\n",
            " [  3 203  17]\n",
            " [  0   7 122]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
           
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "For a RandomForest model with GridSearch:  Precision = 0.925, Recall = 0.938, F1-score = 0.931\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Confusion matrix for SVM model using GridSearch\n",
        "print(confusion_matrix(np.array(y_test), pred_svm_grid))\n",
        "plot_confusion_matrix(svm_grid, X_test, y_test)\n",
        "plt.show()\n",
        "#Finding precision, recall and f-score for SVM (GridSearch) model and updating values in respective lists.\n",
        "precision[4], recall[4], fscore[4], Nil = precision_recall_fscore_support(y_test, pred_svm_grid, average='macro')\n",
        "print('For a SVM model with GridSearch:  Precision = %.3f, Recall = %.3f, F1-score = %.3f'%(precision[4], recall[4], fscore[4]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 403
        },
        "id": "2M3-HaposdQi",
        "outputId": "3d69cfb4-eeb6-49f3-8cbd-21ec85172d5d"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[ 44   4   0]\n",
            " [  4 206  13]\n",
            " [  0   9 120]]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.8/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_confusion_matrix is deprecated; Function `plot_confusion_matrix` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: ConfusionMatrixDisplay.from_predictions or ConfusionMatrixDisplay.from_estimator.\n",
            "  warnings.warn(msg, category=FutureWarning)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 2 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "For a SVM model with GridSearch:  Precision = 0.920, Recall = 0.924, F1-score = 0.922\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating a line plot to compare teh precision, recall and f1-score of all the models.\n",
        "#This graph is plotted to find the best fit for our dataset.\n",
        "plt.figure(figsize=(15,5))\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "             y = precision,label = 'precision')\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "             y = recall,label = 'recall')\n",
        "sns.lineplot(x = ['DecisionTree', 'RandomForest','SVM', 'RandomForest (GridSearch)', 'SVM (GridSearch)'], \n",
        "             y = fscore,label = 'f1-score')\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "id": "yn45FuEDsgGL",
        "outputId": "e3c3a13f-3693-4190-b5f2-be2fa44c58ba"
      },
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1080x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Making Predictions with best fit model"
      ],
      "metadata": {
        "id": "60gzwBRIslfY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Therefore the most accurate results for all the applicants are as follows.\n",
        "possibility = {0: \"High\", 1: \"Moderate\", 2: \"Low\"}\n",
        "print('Applicant\\tLabel\\tPossibility of Dyslexia')\n",
        "print('1\\t\\t{}\\t\\t{}'.format(label_1[3], possibility[label_1[3]]))\n",
        "print('2\\t\\t{}\\t\\t{}'.format(label_2[3], possibility[label_2[3]]))\n",
        "print('3\\t\\t{}\\t\\t{}'.format(label_3[3], possibility[label_3[3]]))\n",
        "print('4\\t\\t{}\\t\\t{}'.format(label_4[3], possibility[label_4[3]]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1BqT3iResiR7",
        "outputId": "278c79b3-91b9-420c-93d9-8a61a2bd47a6"
      },
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Applicant\tLabel\tPossibility of Dyslexia\n",
            "1\t\t2\t\tLow\n",
            "2\t\t2\t\tLow\n",
            "3\t\t1\t\tModerate\n",
            "4\t\t1\t\tModerate\n"
          ]
        }
      ]
    }
  ]
}
