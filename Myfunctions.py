from math import floor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,RobustScaler,StandardScaler,MinMaxScaler
from sklearn.utils.fixes import signature
from sklearn.metrics import f1_score

# Data Engineering ---------------------------------------------------------------------

def pp(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}%'.format(digits, floor(val) / 10 ** digits)

def kde_target2(df,var_name,targetvar='TARGET'):
    # Calculate the correlation coefficient between the new variable and the target
    corr = df[targetvar].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_0 = df.ix[df[targetvar] == 0, var_name].median()
    avg_1 = df.ix[df[targetvar] == 1, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.ix[df[targetvar] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df[targetvar] == 1, var_name], label = 'TARGET == 1')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for Target 1 = %0.4f' % avg_not_repaid)
    print('Median value for Target 0 = %0.4f' % avg_repaid)

def kde_target3(df,var_name,targetvar='TARGET'):
    # Calculate the correlation coefficient between the new variable and the target
    corr = df[targetvar].corr(df[var_name])
    
    # Calculate medians for repaid vs not repaid
    avg_0 = df.ix[df[targetvar] == 0, var_name].median()
    avg_1 = df.ix[df[targetvar] == 1, var_name].median()
    avg_2 = df.ix[df[targetvar] == 2, var_name].median()
    
    plt.figure(figsize = (12, 6))
    
    # Plot the distribution for target == 0 and target == 1 and target == 2
    sns.kdeplot(df.ix[df[targetvar] == 0, var_name], label = 'TARGET == 0')
    sns.kdeplot(df.ix[df[targetvar] == 1, var_name], label = 'TARGET == 1')
    sns.kdeplot(df.ix[df[targetvar] == 2, var_name], label = 'TARGET == 2')
    
    # label the plot
    plt.xlabel(var_name); plt.ylabel('Density'); plt.title('%s Distribution' % var_name)
    plt.legend();
    
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for Target 0 = %0.4f' % avg_0)
    print('Median value for Target 1 = %0.4f' % avg_1)
    print('Median value for Target 2 = %0.4f' % avg_2)
    
def trans_cate(xt):
# for any categorical variable (dtype == object) with 2 unique categories, we will use label encoding, and for any categorical variable with more than 2 unique categories, we will use one-hot encoding.
# For label encoding, we use the Scikit-Learn LabelEncoder and for one-hot encoding, the pandas get_dummies(df) function.
    xtonehot=xt.copy(deep=True)
    le = LabelEncoder()
    le_count = 0
    encodeset=list()
    # Iterate through the columns
    for col in xtonehot:
        if xtonehot[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(xtonehot[col].unique())) <= 2:
                # Train on the training data
                le.fit(xtonehot[col])
                # Transform both training and testing data
                xtonehot[col] = le.transform(xtonehot[col])
                # Keep track of how many columns were label encoded
                le_count += 1
                encodeset.append(xtonehot[col].name)
    print('%d columns were label encoded.' % le_count,encodeset)
    xtonehot=pd.get_dummies(data=xtonehot)
    print(xtonehot.shape)
    return xtonehot

def repmiss(xt,xts):
# Replace missing data in both object, float and int    
# Int and Float with mean
# Categories with most frequent value
# Replace xtest with missing value from xtrain
    xtrain=xt.copy(deep=True)
    xtest=xts.copy(deep=True)
    lnum=xtrain.select_dtypes(include=['float64','int64']).columns.tolist()
    lcat=xtrain.select_dtypes(include=['object']).columns.tolist() 
    avgtrain_num=xtrain.loc[:,lnum].mean()
    mostfreq_cat=xtrain.loc[:,lcat].mode()
    xtrain.loc[:,lnum]=xtrain.loc[:,lnum].fillna(avgtrain_num)
    xtrain.loc[:,lcat]=xtrain.loc[:,lcat].fillna(mostfreq_cat.T.squeeze())
    
    xtest.loc[:,lnum]=xtest.loc[:,lnum].fillna(avgtrain_num)
    xtest.loc[:,lcat]=xtest.loc[:,lcat].fillna(mostfreq_cat.T.squeeze())
    return xtrain,xtest

def showcatediff(xt,tar,topx):
    xtrain=xt.copy(deep=True)
    tab=xt.groupby([tar]).mean()
    tab=tab.transpose()
    tab['Diff']=tab[1]/tab[0]
    tab=tab.replace([np.inf, -np.inf], np.nan)
    tab=tab.dropna(axis=1,how="all").sort_values(by=['Diff'],ascending=False)
    tab.to_csv('diff.csv',index=True)
    f, ax = plt.subplots(figsize=(25, 10))
    sns.barplot(x=tab['Diff'][:topx],y=tab.index[:topx])
    return tab

def showmiss(xt):
# Display missing variable, return full list and a drop list for trim down
    xtrain=xt.copy(deep=True)
    miss = xtrain.isnull().sum().sort_values(ascending=False)
    total = xtrain.isnull().count().sort_values(ascending=False)
    percent = (xtrain.isnull().sum()/xtrain.isnull().count()).sort_values(ascending=False)
    datatype=xtrain.dtypes
    misstable = pd.concat([total, miss, percent, datatype], axis=1, keys=['Total','Miss','Percent','Datatype']).sort_values(by=['Percent'],ascending=False)
    droplistfull=misstable[(misstable.Percent>0)]
    return misstable,droplistfull

def agg_numeric(df, group_var, df_name):
#     """Aggregates the numeric values in a dataframe. This can
#     be used to create features for each instance of the grouping variable.
    
#     Parameters
#     --------
#         df (dataframe): 
#             the dataframe to calculate the statistics on
#         group_var (string): 
#             the variable by which to group df
#         df_name (string): 
#             the variable used to rename the columns
        
#     Return
#     --------
#         agg (dataframe): 
#             a dataframe with the statistics aggregated for 
#             all numeric columns. Each instance of the grouping variable will have 
#             the statistics (mean, min, max, sum; currently supported) calculated. 
#             The columns are also renamed to keep track of features created.
#     """
    # Remove id variables other than grouping variable
    for col in df:
#         if col != group_var and 'SK_ID' in col:
        if col != group_var in col:
            df = df.drop(columns = col)
     
        group_ids = df[group_var]
        numeric_df = df.select_dtypes('number')
        numeric_df[group_var] = group_ids

        # Group by the specified variable and calculate the statistics
        agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

        # Need to create new column names
        columns = [group_var]
    
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    return agg

def agg_cate(df, group_var, df_name):
#       """Computes counts and normalized counts for each observation
#     of `group_var` of each unique category in every categorical variable
#     Parameters
#     --------
#     df : dataframe 
#         The dataframe to calculate the value counts for.
#     group_var : string
#         The variable by which to group the dataframe. For each unique
#         value of this variable, the final dataframe will have one row
#     df_name : string
#         Variable added to the front of column names to keep track of columns
#     Return
#     --------
#     categorical : dataframe
#         A dataframe with counts and normalized counts of each unique category in every categorical variable
#         with one row for every unique value of the `group_var`. 
#     """
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    column_names = []
   # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names
    
    return categorical

# Classification ---------------------------------------------------------------------

def plot_prec_recall_vs_tresh(precision, recall, thresholds):
    plt.plot(thresholds, precision[:-1], 'b--', label='precision')
    plt.plot(thresholds, recall[:-1], 'g--', label = 'recall')
    plt.plot(thresholds, (2*precision[:-1]*recall[:-1])/(precision[:-1]+recall[:-1]), 'r--', label = 'f1')
    plt.xlabel('Threshold')
    plt.legend(loc='upper right')
    plt.ylim([0,1])
    
def plot_prec_recall(precision, recall, thresholds):
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    # Formcheck=pd.DataFrame([precision,recall,thresholds],columns=['Precision','Recall','Thresholds'])
    # prcheck=pd.DataFrame({'Precision': precision, 'Recall':recall,'Thresholds':thresholds})

# Model Validation ---------------------------------------------------------------------    
    
def gb_f1_score(yTest,pred):
    return 'f1', f1_score(yTest, pred,labels=[1,0]), True
    
def get_best_score(grid):
    best_score = np.sqrt(-grid.best_score_)
    print("\n"*2)
    print("*"*50)
    print("Best Score is ",best_score)    
    print("Best Paras Are ",grid.best_params_)
    print("Best Ests Are ",grid.best_estimator_)
    print("*"*50)
    print("\n")
    return best_score

def learningCurve(X, y, Xval, yval, lambda_=0):
#     """
#     Generates the train and cross validation set errors needed to plot a learning curve
#     returns the train and cross validation set errors for a learning curve. 
    
#     In this function, you will compute the train and test errors for
#     dataset sizes from 1 up to m. In practice, when working with larger
#     datasets, you might want to do this in larger intervals.
    
#     Parameters
#     ----------
#     X : array_like
#         The training dataset. Matrix with shape (m x n + 1) where m is the 
#         total number of examples, and n is the number of features 
#         before adding the bias term.
    
#     y : array_like
#         The functions values at each training datapoint. A vector of
#         shape (m, ).
    
#     Xval : array_like
#         The validation dataset. Matrix with shape (m_val x n + 1) where m is the 
#         total number of examples, and n is the number of features 
#         before adding the bias term.
    
#     yval : array_like
#         The functions values at each validation datapoint. A vector of
#         shape (m_val, ).
    
#     lambda_ : float, optional
#         The regularization parameter.
    
#     Returns
#     -------
#     error_train : array_like
#         A vector of shape m. error_train[i] contains the training error for
#         i examples.
#     error_val : array_like
#         A vecotr of shape m. error_val[i] contains the validation error for
#         i training examples.
    
#     Instructions
#     ------------
#     Fill in this function to return training errors in error_train and the
#     cross validation errors in error_val. i.e., error_train[i] and 
#     error_val[i] should give you the errors obtained after training on i examples.
    
#     Notes
#     -----
#     - You should evaluate the training error on the first i training
#       examples (i.e., X[:i, :] and y[:i]).
    
#       For the cross-validation error, you should instead evaluate on
#       the _entire_ cross validation set (Xval and yval).
    
#     - If you are using your cost function (linearRegCostFunction) to compute
#       the training and cross validation error, you should call the function with
#       the lambda argument set to 0. Do note that you will still need to use
#       lambda when running the training to obtain the theta parameters.
    
#     Hint
#     ----
#     You can loop over the examples with the following:
     
#            for i in range(1, m+1):
#                # Compute train/cross validation errors using training examples 
#                # X[:i, :] and y[:i], storing the result in 
#                # error_train[i-1] and error_val[i-1]
#                ....  
#     """
    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
    Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
    error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

    pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
    pyplot.title('Learning curve for linear regression')
    pyplot.legend(['Train', 'Cross Validation'])
    pyplot.xlabel('Number of training examples')
    pyplot.ylabel('Error')
    pyplot.axis([0, 13, 0, 150])

    print('# Training Examples\tTrain Error\tCross Validation Error')
    for i in range(m):
        print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))     

    # =============================================================
    return error_train, error_val

def plot_validation_curve(estimator, X, y, title=None,ylim=None,param_name=None,
                          param_range=[1,100,1000,10000],cv=10,scoring=None,n_jobs=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores, test_scores = validation_curve(estimator
        , X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(-0.1, 1.1)
    lw = 2
#     plt.grid()
#     plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.01,
#                      color="r")
#     plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.01, color="g")
#     plt.plot(param_range, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(param_range, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#     plt.legend(loc="best")
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,scoring=None,
                        n_jobs=None, train_sizes=[0.1,0.5,1.0]):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, scoring=scoring, train_sizes=train_sizes)
#     train_scores=-1*train_scores
#     test_scores=-1*test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#     """
#     Generate a simple plot of the test and training learning curve.
#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.
#     title : string
#         Title for the chart.
#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.
#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.
#     cv : int, cross-validation generator or an iterable, optional
#         Determines the cross-validation splitting strategy.
#         Possible inputs for cv are:
#           - None, to use the default 3-fold cross-validation,
#           - integer, to specify the number of folds.
#           - :term:`CV splitter`,
#           - An iterable yielding (train, test) splits as arrays of indices.
#         For integer/None inputs, if ``y`` is binary or multiclass,
#         :class:`StratifiedKFold` used. If the estimator is not a classifier
#         or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
#         Refer :ref:`User Guide <cross_validation>` for the various
#         cross-validators that can be used here.
#     n_jobs : int or None, optional (default=None)
#         Number of jobs to run in parallel.
#         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
#         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
#         for more details.
#     train_sizes : array-like, shape (n_ticks,), dtype float or int
#         Relative or absolute numbers of training examples that will be used to
#         generate the learning curve. If the dtype is float, it is regarded as a
#         fraction of the maximum size of the training set (that is determined
#         by the selected validation method), i.e. it has to be within (0, 1].
#         Otherwise it is interpreted as absolute sizes of the training sets.
#         Note that for classification the number of samples usually have to
#         be big enough to contain at least one sample from each class.
#         (default: np.linspace(0.1, 1.0, 5))
#     """

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return