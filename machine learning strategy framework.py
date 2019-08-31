# -*- coding:utf-8 -*-

from CloudQuant import SDKCoreEngine  # 导入量子金服SDK
from CloudQuant import AssetType
from CloudQuant import QuoteCycle
from CloudQuant import OrderType
import numpy as np  # 使用numpy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import neural

np.seterr(invalid='ignore')


OVERRETURN = 0 #设定分类阈值，这里设为0
TARGET_DRAWDOWN=0.1

config = {
    'username': 'ldy123',
    'password': 'qdldy123',
    'rootpath': 'D:\quantcloud\QuantDesk',  # 客户端所在路径
    'assetType': AssetType.Stock,
    'initCapitalStock': 100000000,  # 初始资金
    'startDate': 20150101,  # 交易开始日期
    'endDate': 20180420,  # 交易结束日期
    'cycle': QuoteCycle.D,  # 回放粒度为日线
    'feeRate': 0.001,
    'feeLimit': 5,
    'strategyName': 'LR123',  # 策略名
    "logfile": "LR.log",
    'dealByVolume': False
}


def initial(sdk):
    day_list=sdk.getDayList() #获取所有交易日
    day_list=[i for i in day_list if i>=config['startDate']] #获取起始日期之后的交易日
    # 获取每月第一个交易日列表
    day_monthly=[]
    for i in range(len(day_list)-1):
        if i==0:
            day_monthly.append(day_list[i])
        elif day_list[i+1]//100!=day_list[i]//100:
            day_monthly.append(day_list[i+1])
    sdk.setGlobal("day_list",day_list)
    sdk.setGlobal('day_monthly',day_monthly)
    sdk.setGlobal("a_r", [])

def initPerDay(sdk):
    pass

def strategy(sdk):
    if sdk.isAssetTradingTime(AssetType.Stock):
        today = sdk.getNowDate()
        day_monthly = sdk.getGlobal("day_monthly")
        day_list = sdk.getGlobal("day_list")
        asset_record = sdk.getGlobal('a_r')
        if today in day_monthly:
            asset_record = []
        asset_current = sdk.getAccountAsset().currentAsset
        marketValue_current = sdk.getAccountAsset().currentMarketValue
        asset_record.append(asset_current)
        asset_max = max(asset_record)
        draw_down = 1 - asset_current / asset_max
        target_position = min(max(1 - draw_down / TARGET_DRAWDOWN, 0), 1)
        current_position = marketValue_current / asset_current
        if current_position > target_position:
            reduction_percent = (current_position - target_position) / current_position
            pos = sdk.getPositions()
            stock_owned = [i.code for i in pos]
            quote = sdk.getQuotes(stock_owned)
            sellOrder = []
            for item in pos:
                stock = item.code
                q = quote[stock]
                volume = item.optPosition * reduction_percent // 100 * 100
                if q:
                    sellOrder.append([stock, q.current, volume, -1])
                    sdk.makeOrders(sellOrder)
        if current_position < target_position:
            asset_current = sdk.getAccountAsset().currentAsset
            budget = asset_current * (target_position - current_position)

        if today == day_monthly[0]:  # 回测第一个交易日，不交易
            return
        if today in day_monthly:  # 如果交易日为每月第一天，进行交易
            lastday = day_monthly[day_monthly.index(today) - 1]  # 获取上个月第一天
            delta = day_list.index(today) - day_list.index(lastday)  # 获取周期交易日长度
            FACTORS_TRAIN = []  # 存训练数据
            FACTORS = []  # 存预测数据
            # 取10个因子
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'TMV')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'FMV')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'STOCKZF22')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'STOCKPJCJ5/60')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'HSL22')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'SALESGROWRATE1')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'BP')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'PROFITGROWRATE1')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'SP')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'SALESGROWRATE')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'ROA')
            getFactors(sdk, FACTORS_TRAIN, FACTORS, delta, 'D2A')
            # 删去NaN项
            condition1 = getCondition(FACTORS_TRAIN)
            condition2 = getCondition(FACTORS)
            condition = reduce(lambda a, b: np.logical_and(a, b), [condition1, condition2])
            # 标准化
            FACTORS_TRAIN = normalize(FACTORS_TRAIN, condition)
            FACTORS = normalize(FACTORS, condition)
            FACTORS_TRAIN = transposeData(FACTORS_TRAIN)
            FACTORS = transposeData(FACTORS)
            # 计算上月Benchmark（沪深300）收益率
            indexData = sdk.getFactorData('LZ_CN_STKA_INDXQUOTE_CLOSE')
            benchmark_column = 19
            benchmark_last = indexData[-delta, benchmark_column]
            benchmark_now = indexData[-1, benchmark_column]
            benchmark_rate = (benchmark_now - benchmark_last) / benchmark_last
            # 计算上月A股全体个股收益率
            stockData = sdk.getFactorData("LZ_CN_STKA_QUOTE_TCLOSE")
            stock_last = stockData[-delta, :][condition]
            stock_now = stockData[-1, :][condition]
            target = getTarget(stock_now, stock_last, benchmark_rate)
            """
            # Using Logistic Regression
            classifier = LogisticRegression(C = 0.001)
            classifier.fit(FACTORS_TRAIN, target)  # 训练模型
            predict_prob = classifier.predict_proba(FACTORS)  # 用模型预测
            predict_prob = np.transpose(predict_prob)[1]
            """
            # Using Neural Network

            parameters = neural.nn_model(FACTORS_TRAIN.T, target.reshape(target.shape[0], 1).T, n_h = 20, num_iterations = 10000, print_cost=False)
            predict_prob = neural.predict(parameters, FACTORS.T)

            # 选取概率最大的30支股票
            threshold = np.sort(predict_prob)[::-1][30]
            predict = np.array(predict_prob > threshold)
            """
            stocklist = sdk.getStockList()
            for i in predict:
                if predict[i] == True:
                    print stocklist[i]
                else:
                    pass
            """
            trade(sdk, budget, condition, predict)  # 调仓交易

        sdk.setGlobal("a_r", asset_record)




def getCondition(FACTORS): # 删除所有含NaN数据
    conditions=[]
    for factor in FACTORS:
        conditions.append(~np.isnan(factor))
    condition = reduce(lambda a, b: np.logical_and(a, b), conditions)
    return condition

def normalize(FACTORS,condition): # 标准化
    FACTORS_SCALED=[]
    for factor in FACTORS:
        factor = factor[condition]
        factor = scale(factor)
        FACTORS_SCALED.append(factor)
    return FACTORS_SCALED

def transposeData(FACTORS): #转置输入数据
    FACTORS=np.array(FACTORS)
    data=np.transpose(FACTORS)
    return data

def getTarget(stock_today,stock_last,benchmark_rate): #给训练数据分类，赋予label
    n=len(stock_today)
    target=[False]*n
    for i in range(n):
        rate=(stock_today[i]-stock_last[i])/stock_last[i]
        if rate>=OVERRETURN+benchmark_rate:
            target[i]=True
    target=np.array(target)
    return target

def trade(sdk,budget,condition,predict): #调仓命令：卖出不在股票池中的股票，将所有资产平均买入在股票池中的股票
    stock_list=np.array(sdk.getStockList())
    stock_list=stock_list[condition]
    sellList=stock_list[~predict]
    sellList=list(sellList)
    if sellList>0: #卖出应售股票
        for stock in sellList:
            pos=sdk.getPositions(code=stock)
            if pos:
                quote=sdk.getQuote(stock)
                if quote:
                    sdk.makeOrder(stock,quote.current,pos[0].optPosition,-1)
                    sdk.sdklog([stock,quote.current,pos[0].optPosition,-1],'sell')

    buyList=stock_list[predict]
    buyList=list(buyList)

    if buyList: #买入应购股票
        budget=budget/len(buyList)
        for stock in buyList:
            quote = sdk.getQuote(stock)
            if quote:
                volume = (int(budget /(quote.current*(1+config['feeRate'])))//100)*100
                sdk.makeOrder(stock, quote.current, volume, 1)
                sdk.sdklog([stock, quote.current, volume, 1], 'buy')

def getFactors(sdk,FACTORS_TRAIN,FACTORS,delta,factor_name): #获取当日和上一个周期第一天的factor数据，存入FACTORS_TRAIN和FACTORS
    if factor_name=='TMV':
        TMA_TRAIN = sdk.getFactorData("LZ_CN_STKA_VAL_A_TCAP")[-delta, :]  # 因子1，总市值
        FACTORS_TRAIN.append(TMA_TRAIN)
        TMA = sdk.getFactorData("LZ_CN_STKA_VAL_A_TCAP")[-1, :]  # 因子1，总市值
        FACTORS.append(TMA)

    if factor_name=="FMV":
        FMA_TRAIN = sdk.getFactorData("LZ_CN_STKA_VAL_A_FCAP")[-delta, :]  # 因子2，流通市值
        FACTORS_TRAIN.append(FMA_TRAIN)
        FMA = sdk.getFactorData("LZ_CN_STKA_VAL_A_FCAP")[-1, :]  # 因子2，流通市值
        FACTORS.append(FMA)

    if factor_name=='STOCKZF22':
        stockData = sdk.getFactorData("LZ_CN_STKA_QUOTE_TCLOSE")
        STOCKZF22_TRAIN_last = stockData[-delta - 21, :]
        STOCKZF22_TRAIN_now = stockData[-delta, :]
        STOCKZF22_TRAIN = (STOCKZF22_TRAIN_now - STOCKZF22_TRAIN_last) / STOCKZF22_TRAIN_last
        FACTORS_TRAIN.append(STOCKZF22_TRAIN)
        stockData = sdk.getFactorData("LZ_CN_STKA_QUOTE_TCLOSE")
        STOCKZF22_last = stockData[-1 - 21, :]
        STOCKZF22_now = stockData[-1, :]
        STOCKZF22 = (STOCKZF22_now - STOCKZF22_last) / STOCKZF22_last
        FACTORS.append(STOCKZF22)

    if factor_name=="STOCKPJCJ5/60":
        VWAP5=sdk.getFactorData("LZ_CN_STKA_DERI_VWAP_5")
        VWAP60=sdk.getFactorData("LZ_CN_STKA_DERI_VWAP_60")
        VWAP5=VWAP5.astype(np.float64)
        VWAP60 = VWAP60.astype(np.float64)
        STOCKPJCJ_TRAIN=VWAP5[-delta,:]/VWAP60[-delta,:]
        FACTORS_TRAIN.append(STOCKPJCJ_TRAIN)
        STOCKPJCJ = VWAP5[-1, :] / VWAP60[-1, :]
        FACTORS.append(STOCKPJCJ)

    if factor_name=="HSL22":
        HSLDATA=sdk.getFactorData("LZ_CN_STKA_VAL_TURN")
        HSL_TRAIN=np.nanmean(HSLDATA[-delta-21:-delta+1,:],axis=0)
        FACTORS_TRAIN.append(HSL_TRAIN)
        HSL=np.nanmean(HSLDATA[-22:,:],axis=0)
        FACTORS.append(HSL)

    if factor_name=="SALESGROWRATE1":
        DATA=sdk.getFactorData("LZ_CN_STKA_FIN_IND_QFA_YOYSALES")
        TRAIN=DATA[-delta,:]
        FACTORS_TRAIN.append(TRAIN)
        PREDICT=DATA[-1,:]
        FACTORS.append(PREDICT)

    if factor_name=="BP":
        BPDATA=sdk.getFactorData("LZ_CN_STKA_VAL_PB")
        BP_TRAIN=1/BPDATA[-delta,:]
        FACTORS_TRAIN.append(BP_TRAIN)
        BP=1/BPDATA[-1,:]
        FACTORS.append(BP)

    if factor_name=="PROFITGROWRATE1":
        DATA=sdk.getFactorData("LZ_CN_STKA_FIN_IND_QFA_YOYPRFT")
        TRAIN = DATA[-delta, :]
        FACTORS_TRAIN.append(TRAIN)
        PREDICT = DATA[-1, :]
        FACTORS.append(PREDICT)

    if factor_name=="SP":
        DATA = sdk.getFactorData("LZ_CN_STKA_DERI_SP_TTM")
        TRAIN = DATA[-delta, :]
        TRAIN = TRAIN.astype(np.float64)
        FACTORS_TRAIN.append(TRAIN)
        PREDICT = DATA[-1, :]
        PREDICT = PREDICT.astype(np.float64)
        FACTORS.append(PREDICT)

    if factor_name=="SALESGROWRATE":
        DATA=sdk.getFactorData("LZ_CN_STKA_FIN_IND_YOY_OR")
        TRAIN = DATA[-delta, :]
        FACTORS_TRAIN.append(TRAIN)
        PREDICT = DATA[-1, :]
        FACTORS.append(PREDICT)

    if factor_name=="ROA":
        DATA=sdk.getFactorData("LZ_CN_STKA_FIN_IND_ROA")
        TRAIN = DATA[-delta, :]
        FACTORS_TRAIN.append(TRAIN)
        PREDICT = DATA[-1, :]
        FACTORS.append(PREDICT)

    if factor_name=="D2A":
        DATA=sdk.getFactorData("LZ_CN_STKA_FIN_IND_DEBTTOASTS")
        TRAIN = DATA[-delta, :]
        FACTORS_TRAIN.append(TRAIN)
        PREDICT = DATA[-1, :]
        FACTORS.append(PREDICT)

def main():
    # 将策略函数加入
    config['initial'] = initial
    config['strategy'] = strategy
    config['preparePerDay'] = initPerDay
    # 启动SDK
    SDKCoreEngine(**config).run()


if __name__ == "__main__":
    main()
