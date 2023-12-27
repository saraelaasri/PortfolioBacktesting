from flask import Flask, render_template, request, jsonify, session
from register import reg
import os
from pypfopt import expected_returns, risk_models
from model import create_filtered_prices_graph,get_filtered_prices, dtrgraph, describe_prices,plot_correlation_heatmap, display_calculated_ef_with_random, generate_weights_table, generate_weights_plot, perfermance,backtest_markowitz_portfolio_img,calculate_max_drawdown,backtest_markowitz_portfolio,backtest_black_litterman_portfolios,backtest_black_litterman_portfolios_imge,backtest_genetic_portfolio,backtest_genetic_portfolio_image,run_backtests
import yfinance as yf
import numpy as np
from datetime import timedelta
app = Flask(__name__)
app.secret_key = os.urandom(32)

app.register_blueprint(reg)
@app.route('/',methods=['GET','POST'])
def index():  # put application's code here
    if request.method == 'POST':
        actifs = request.form['actifs']
        listactifs=actifs.split()
        datedebut = request.form['datedebut']
        datefin = request.form['datefin']
        num_prtfolio = request.form['num_portfolio']
        print("listactifs--> ", listactifs, "date debut--> ", datedebut, "num_prtfolio--> ",num_prtfolio)

        population=request.form['population']
        generation=request.form['generation']
        mutation=request.form['mutation']
        ellistime=request.form['ellistime']
        risk_free_rate=request.form['risk_free_rate']
        vue1=request.form['vue1']
        vue2=request.form['vue2']
        vue3=request.form['vue3']
        vue4=request.form['vue4']
        vue5=request.form['vue5']
        list=[]
        list.append(float(vue1))
        list.append(float(vue2))
        list.append(float(vue3))
        list.append(float(vue4))
        list.append(float(vue5))
        print("hadi liste des vues ",list)
        my_array = np.array(list)
        print('np-array')
        print(my_array)
        #function 1 : courbe1 de prix
        dataa=get_filtered_prices(listactifs, datedebut, datefin)
        image64 = create_filtered_prices_graph(dataa)
        print("image64")
        print(image64)
        #courbe 2
        dtr = dtrgraph(dataa)
        print("IMAAAAGE")
        print(dtr)
        #Table Statistique des rendements
        table= describe_prices(dataa)
        print(table)
        correlation=plot_correlation_heatmap(dataa)
        #Mean Variance
        mean_varianceImg=display_calculated_ef_with_random(dataa,int(num_prtfolio))
        print("pop: ",population, generation, mutation, ellistime, vue1, vue2, vue3,vue4,vue5)
        tpoids=generate_weights_table(dataa,int(num_prtfolio),my_array,int(population),int(generation),float(mutation),float(ellistime), float(risk_free_rate), tau_range=np.linspace(0.001, 1, 100))
        print(mean_varianceImg)
        plot_image=generate_weights_plot(tpoids)
        tperformance=perfermance(tpoids,dataa,float(risk_free_rate))
        image_mv=backtest_markowitz_portfolio_img(dataa)
        print(image_mv)
        image_bl=backtest_black_litterman_portfolios_imge(dataa, my_array)
        image_gen=backtest_genetic_portfolio_image(dataa,int(population),int(generation), float(mutation), float(ellistime), float(risk_free_rate))
        backtest=run_backtests(dataa,my_array,int(population),int(generation), float(mutation), float(ellistime), float(risk_free_rate))
        response ={'image64': image64,
                   "table": table.to_html(),
                   'correlation': correlation,
                   'mean_varianceImg': mean_varianceImg,
                   'tpoids': tpoids.to_html(),
                   'plot_image': plot_image,
                   'tperformance': tperformance.to_html(),
                   'image_mv':image_mv,
                    'image_bl':image_bl,
                   'image_gen':image_gen,
                   'backtest':backtest.to_html()
                   }
        return jsonify(response)
    return render_template("index.html")


@app.route("/optimisation")
def about_page():
    return render_template("optimisation.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

if __name__ == '__main__':
    app.run(debug=True)
