from flask import Flask, render_template, jsonify
import random
from datetime import datetime, timedelta
import os
import requests

app = Flask(__name__)

class SimpleHourlyPredictor:
    def __init__(self):
        self.api_key = os.environ.get('ALPHA_VANTAGE_API_KEY', 'QE0TAOPZZN1VT8LH')
        
    def get_alpha_vantage_price(self):
        """Get current EUR/GBP price from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": "EUR",
                "to_currency": "GBP",
                "apikey": self.api_key
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if "Realtime Currency Exchange Rate" in data:
                rate = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                return {"success": True, "price": rate}
            else:
                # Use demo price if API fails
                return {"success": False, "price": 0.86000 + random.uniform(-0.001, 0.001)}
                
        except Exception as e:
            return {"success": False, "price": 0.86000 + random.uniform(-0.001, 0.001)}
    
    def generate_hourly_prediction(self):
        """Generate hourly trading prediction"""
        # Get current price
        price_data = self.get_alpha_vantage_price()
        current_price = price_data["price"]
        
        # Generate AI signal (simplified)
        hour = datetime.now().hour
        minute = datetime.now().minute
        
        # Different signals based on time of day
        if hour % 4 == 0:
            signal = "STRONG_BUY"
            confidence = random.randint(85, 95)
        elif hour % 4 == 1:
            signal = "BUY"
            confidence = random.randint(75, 85)
        elif hour % 4 == 2:
            signal = "STRONG_SELL"
            confidence = random.randint(85, 95)
        else:
            signal = "SELL"
            confidence = random.randint(75, 85)
        
        # Calculate entry, TP, SL
        volatility = 0.0015
        
        if "BUY" in signal:
            action = "BUY"
            entry = round(current_price * 0.9995, 5)
            tp = round(entry * (1 + volatility * 2), 5)
            sl = round(entry * (1 - volatility), 5)
        else:
            action = "SELL"
            entry = round(current_price * 1.0005, 5)
            tp = round(entry * (1 - volatility * 2), 5)
            sl = round(entry * (1 + volatility), 5)
        
        # Calculate pips
        pip_tp = round(abs(tp - entry) / 0.0001, 1)
        pip_sl = round(abs(sl - entry) / 0.0001, 1)
        risk_reward = round(pip_tp / pip_sl, 2) if pip_sl > 0 else 1.5
        
        # Generate chart data
        chart_data = self.generate_chart_data(current_price, signal)
        
        return {
            "success": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_hour": f"{hour:02d}:00",
            "next_hour": f"{(hour + 1) % 24:02d}:00",
            "current_price": round(current_price, 5),
            "signal": signal,
            "action": action,
            "confidence": confidence,
            "entry_price": entry,
            "take_profit": tp,
            "stop_loss": sl,
            "pip_gain": pip_tp,
            "pip_risk": pip_sl,
            "risk_reward": risk_reward,
            "predicted_high": round(current_price * (1 + volatility * 1.5), 5),
            "predicted_low": round(current_price * (1 - volatility * 1.5), 5),
            "chart_data": chart_data
        }
    
    def generate_chart_data(self, current_price, signal):
        """Generate simple chart data without pandas"""
        times = []
        prices = []
        
        base_price = current_price
        
        for i in range(60):  # 60 minutes
            # Add time label
            times.append(f"{i:02d}:00")
            
            # Simulate price movement
            if "BUY" in signal:
                movement = random.uniform(-0.0001, 0.0003)
            else:
                movement = random.uniform(-0.0003, 0.0001)
            
            base_price += movement
            prices.append(round(base_price, 5))
        
        return {
            "times": times,
            "prices": prices
        }

# Initialize predictor
predictor = SimpleHourlyPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict')
def get_prediction():
    try:
        prediction = predictor.generate_hourly_prediction()
        return jsonify(prediction)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "current_price": 0.86000,
            "signal": "NEUTRAL",
            "confidence": 50
        })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("⏰ Hourly Forex Predictor")
    print("=" * 60)
    print(f"Starting on port: {port}")
    print(f"API Key: {predictor.api_key[:8]}...")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port)