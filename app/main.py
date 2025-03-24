from app import app

if __name__ == '__main__':
    # Запуск Flask-сервера
    app.run(host='0.0.0.0', port=5000, debug=True)
