<a href="https://www.npmjs.com/package/vue"><img src="https://img.shields.io/npm/l/vue.svg?sanitize=true" alt="License"></a>



# USD/PLN price prediction
This project aims to predict the USD/PLN price using Linear Regression and Recurrent Neural Networks implemented in Python.

### Preparation

1. Clone repository `git clone git@github.com:timschopinski/dolar-price-prediction.git` 

    (or with HTTPS `git clone https://github.com/timschopinski/dolar-price-prediction.git`)
 
2. Create virtualenv `python -m venv venv`
3. Upgrade setup tools `pip install --upgrade pip setuptools wheel`
4. Activate env and install libraries `pip install -r requirements.txt`
5. go to ./app and execute `python manage.py save_chart`



### Running with docker-compose 

1. Clone repository `git clone git@github.com:timschopinski/dolar-price-prediction.git` 

    (or with HTTPS `git clone https://github.com/timschopinski/dolar-price-prediction.git`)
 
2. Install [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) and [docker-compose](https://docs.docker.com/compose/install/).
3. Run `docker compose run --rm app sh -c "python manage.py save_chart"`

