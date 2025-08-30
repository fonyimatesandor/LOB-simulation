from sortedcontainers import SortedDict
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class OrderBook:
    def __init__(self, depth, messages_path, orderbook_path):
        self.depth = depth
        self.messages = pd.read_csv(messages_path, names=[
                                    'Time (sec)', 'Event type', 'Order ID', 'Size', 'Price', 'Direction'])
        self.orderbook_official = pd.read_csv(orderbook_path, names=[f"{side} {typ} {i+1}" for i in range(
            depth) for side, typ in [("Ask", "Price"), ("Ask", "Size"), ("Bid", "Price"), ("Bid", "Size")]])

        self.bids = SortedDict()
        self.asks = SortedDict()

        self.parse_base_order()

    def initialize(self, k):
        self.process_messages(k)
        self.calculate_metrics(k)
        self.filter_same_time_data()
        self.calculate_interpolators()

    def process_messages(self, k):
        self.parse_base_order()
        orderbook_array = np.zeros((k, self.depth * 4))
        self.times = np.zeros(k)
        for i in range(k):
            message = self.messages.iloc[i]
            self.times[i] = message['Time (sec)']
            self.process_message(message)
            self.correct_shorter_orderbook(i)
            self.correct_orderbook(i)

            orderbook_array[i,:] = self.get_orderbook_list()

        self.orderbook_calculated = pd.DataFrame(orderbook_array, columns = self.orderbook_official.columns)

        return self.validate_orderbook(k-1)


    def process_message(self, message):
        event_type = message['Event type']
        size = message['Size']
        price = message['Price']
        direction = message['Direction']

        if event_type == 1:
            if direction == 1:
                self.bids[price] = self.bids.get(price, 0) + size
            else:
                self.asks[price] = self.asks.get(price, 0) + size
        elif event_type == 2 or event_type == 3:
            if direction == 1:
                if price in self.bids:
                    self.bids[price] -= size
                    if self.bids[price] <= 0:
                        del self.bids[price]
            else:
                if price in self.asks:
                    self.asks[price] -= size
                    if self.asks[price] <= 0:
                        del self.asks[price]
        elif event_type == 4 or event_type == 5:
            if direction == 1:
                if price in self.bids:
                    self.bids[price] -= size
                    if self.bids[price] <= 0:
                        del self.bids[price]
            else:
                if price in self.asks:
                    self.asks[price] -= size
                    if self.asks[price] <= 0:
                        del self.asks[price]


    def calculate_metrics(self, k):
        self.parse_base_order()
        if self.orderbook_calculated is None or len(self.orderbook_calculated) < k:
            self.process_messages(k)

        
        self.metrics_calculated = pd.DataFrame(columns = [
            'Time (sec)', 'Mid Price', 'Weighted Mid Price', 'L10 Weighted Mid Price', 'Microprice',
            'L1 Imbalance', 'L10 Cumulative Imbalance', 'VWAP Imbalance', 'Order Flow Imbalance', 
            'Slope Bid', 'Slope Ask', 'Entropy Bid', 'Entropy Ask', 'HHI Bid', 'HHI Ask'
        ])

        self.metrics_calculated['Time (sec)'] = self.times
        self.metrics_calculated['Mid Price'] = self.orderbook_calculated.apply(self.mid_price, axis=1)
        self.metrics_calculated['Weighted Mid Price'] = self.orderbook_calculated.apply(self.weighted_mid_price, axis=1)
        self.metrics_calculated['L10 Weighted Mid Price'] = self.orderbook_calculated.apply(self.l10_weighted_mid_price, axis=1)
        self.metrics_calculated['Microprice'] = self.orderbook_calculated.apply(self.microprice, axis=1)
        self.metrics_calculated['L1 Imbalance'] = self.orderbook_calculated.apply(self.l1_imbalance, axis=1)
        self.metrics_calculated['L10 Cumulative Imbalance'] = self.orderbook_calculated.apply(self.l10_cumulative_imbalance, axis=1)
        self.metrics_calculated['VWAP Imbalance'] = self.orderbook_calculated.apply(self.vwap_imbalance, axis=1)
        self.metrics_calculated['Order Flow Imbalance'] = self.orderbook_calculated.apply(self.order_flow_imbalance, axis=1)
        self.metrics_calculated[['Slope Bid', 'Slope Ask']] = pd.DataFrame(self.orderbook_calculated.apply(self.calculate_slope, axis=1).tolist(), index=self.orderbook_calculated.index)
        self.metrics_calculated[['Entropy Bid', 'Entropy Ask']] = pd.DataFrame(self.orderbook_calculated.apply(self.entropy, axis=1).tolist(), index=self.orderbook_calculated.index)
        self.metrics_calculated[['HHI Bid', 'HHI Ask']] = pd.DataFrame(self.orderbook_calculated.apply(self.hhi, axis=1).tolist(), index=self.orderbook_calculated.index)

    def order_flow_imbalance(self, row):
        idx = row.name
        if idx == 0:
            return 0
        prev_row = self.orderbook_calculated.iloc[idx - 1]
        curr_row = row

        prev_bids = [prev_row[f"Bid Size {i+1}"] for i in range(self.depth)]
        prev_asks = [prev_row[f"Ask Size {i+1}"] for i in range(self.depth)]
        curr_bids = [curr_row[f"Bid Size {i+1}"] for i in range(self.depth)]
        curr_asks = [curr_row[f"Ask Size {i+1}"] for i in range(self.depth)]

        bid_flow = np.sum(np.array(curr_bids) - np.array(prev_bids))
        ask_flow = np.sum(np.array(curr_asks) - np.array(prev_asks))
        if (bid_flow + ask_flow) == 0:
            return 0
        return (bid_flow - ask_flow) / (bid_flow + ask_flow)
    
    def mid_price(self, row):
        ask_price = row[f"Ask Price 1"]
        bid_price = row[f"Bid Price 1"]
        return (bid_price + ask_price) / 2
    
    def weighted_mid_price(self, row):
        bid_price = row[f"Bid Price 1"]
        bid_size = row[f"Bid Size 1"]
        ask_price = row[f"Ask Price 1"]
        ask_size = row[f"Ask Size 1"]
        return (bid_price * bid_size + ask_price * ask_size) / (bid_size + ask_size)
     
    def l10_weighted_mid_price(self, row):
        bid_prices = [row[f"Bid Price {i+1}"] for i in range(self.depth)]
        bid_sizes = [row[f"Bid Size {i+1}"] for i in range(self.depth)]
        ask_prices = [row[f"Ask Price {i+1}"] for i in range(self.depth)]
        ask_sizes = [row[f"Ask Size {i+1}"] for i in range(self.depth)]

        total_value = sum(p * s for p, s in zip(bid_prices, bid_sizes)) + sum(p * s for p, s in zip(ask_prices, ask_sizes))
        total_size = sum(bid_sizes) + sum(ask_sizes)
        return total_value / total_size if total_size != 0 else np.nan
        
    def microprice(self, row):
        return (row[f"Bid Price 1"] * row[f"Ask Size 1"] + row[f"Ask Price 1"] * row[f"Bid Size 1"]) / (row[f"Bid Size 1"] + row[f"Ask Size 1"])

    def l1_imbalance(self, row):
        return (row[f"Bid Size 1"] - row[f"Ask Size 1"]) / (row[f"Bid Size 1"] + row[f"Ask Size 1"])

    def l10_cumulative_imbalance(self, row):
        return (sum(row[f"Bid Size {i+1}"] for i in range(self.depth)) - sum(row[f"Ask Size {i+1}"] for i in range(self.depth))) / (sum(row[f"Bid Size {i+1}"] for i in range(self.depth)) + sum(row[f"Ask Size {i+1}"] for i in range(self.depth)))

    def vwap_imbalance(self, row):
        bid_vwap = sum(price * size for price, size in zip((row[f"Bid Price {i+1}"] for i in range(self.depth)), (row[f"Bid Size {i+1}"] for i in range(self.depth)))) / sum(row[f"Bid Size {i+1}"] for i in range(self.depth))
        ask_vwap = sum(price * size for price, size in zip((row[f"Ask Price {i+1}"] for i in range(self.depth)), (row[f"Ask Size {i+1}"] for i in range(self.depth)))) / sum(row[f"Ask Size {i+1}"] for i in range(self.depth))
        return (bid_vwap - ask_vwap) / (bid_vwap + ask_vwap)

    def calculate_slope(self, row):
        prices_bid = [row[f"Bid Price {i+1}"] for i in range(self.depth)]
        volumes_bid = [row[f"Bid Size {i+1}"] for i in range(self.depth)]

        prices_ask = [row[f"Ask Price {i+1}"] for i in range(self.depth)]
        volumes_ask = [row[f"Ask Size {i+1}"] for i in range(self.depth)]

        if len(prices_bid) >= 2 and len(volumes_bid) >= 2:
            slope_bid, _ = np.polyfit(prices_bid, volumes_bid, 1)
        else:
            slope_bid = np.nan

        if len(prices_ask) >= 2 and len(volumes_ask) >= 2:
            slope_ask, _ = np.polyfit(prices_ask, volumes_ask, 1)
        else:
            slope_ask = np.nan

        return slope_bid, slope_ask
    
    def entropy(self, row):
        bid_sizes = [row[f"Bid Size {i+1}"] for i in range(self.depth)]
        ask_sizes = [row[f"Ask Size {i+1}"] for i in range(self.depth)]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)

        volumes_bid = [size / total_bid for size in bid_sizes if total_bid > 0]
        volumes_ask = [size / total_ask for size in ask_sizes if total_ask > 0]

        entropy_bid = -sum(p * np.log(p) for p in volumes_bid if p > 0)
        entropy_ask = -sum(p * np.log(p) for p in volumes_ask if p > 0)

        return entropy_bid, entropy_ask
    def hhi(self, row):
        bid_sizes = [row[f"Bid Size {i+1}"] for i in range(self.depth)]
        ask_sizes = [row[f"Ask Size {i+1}"] for i in range(self.depth)]

        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)

        volumes_bid = [size / total_bid for size in bid_sizes if total_bid > 0]
        volumes_ask = [size / total_ask for size in ask_sizes if total_ask > 0]

        hhi_bid = sum(p ** 2 for p in volumes_bid)
        hhi_ask = sum(p ** 2 for p in volumes_ask)

        return hhi_bid, hhi_ask

    def correct_shorter_orderbook(self, k):
        if len(self.bids) < self.depth:
            for j in range(self.depth):
                if self.orderbook_official.iloc[k][f"Bid Price {j+1}"] not in self.bids:
                    self.bids[self.orderbook_official.iloc[k]
                              [f"Bid Price {j+1}"]] = self.orderbook_official.iloc[k][f"Bid Size {j+1}"]

        if len(self.asks) < self.depth:
            for j in range(self.depth):
                if self.orderbook_official.iloc[k][f"Ask Price {j+1}"] not in self.asks:
                    self.asks[self.orderbook_official.iloc[k]
                              [f"Ask Price {j+1}"]] = self.orderbook_official.iloc[k][f"Ask Size {j+1}"]

    def correct_orderbook(self, k):
        if not self.validate_orderbook(k):
            correction_message = self.generate_correction_message(k)
            self.process_message(correction_message)

    def generate_correction_message(self, k):
        current_orderbook = self.get_orderbook_list()
        official_orderbook = self.orderbook_official.iloc[k].tolist()
        correction_message = {}

        for j in range(0, len(current_orderbook)):
            if current_orderbook[j] != official_orderbook[j] and j % 2 == 1:
                correction_message['Event type'] = 1
                correction_message['Order ID'] = 0
                correction_message['Size'] = official_orderbook[j] - \
                    current_orderbook[j]
                correction_message['Price'] = official_orderbook[j-1]
                correction_message['Direction'] = - \
                    1 if j % 4 == 0 or j % 4 == 1 else 1
                break
            elif current_orderbook[j] != official_orderbook[j] and j % 2 == 0:
                correction_message['Event type'] = 1
                correction_message['Order ID'] = 0
                correction_message['Size'] = official_orderbook[j+1]
                correction_message['Price'] = official_orderbook[j]
                correction_message['Direction'] = - \
                    1 if j % 4 == 0 or j % 4 == 1 else 1
                break

        return correction_message

    def parse_base_order(self):
        self.bids.clear()
        self.asks.clear()

        first_row = self.orderbook_official.iloc[0]

        for i in range(self.depth):
            ask_price = first_row[f"Ask Price {i+1}"]
            ask_size = first_row[f"Ask Size {i+1}"]
            bid_price = first_row[f"Bid Price {i+1}"]
            bid_size = first_row[f"Bid Size {i+1}"]

            self.asks[ask_price] = ask_size
            self.bids[bid_price] = bid_size

    def get_orderbook_list(self):
        row = []
        for _ in range(self.depth):
            ask_price = list(self.asks.keys())[_]
            ask_size = self.asks[ask_price]
            bid_price = list(self.bids.keys())[-(_+1)]
            bid_size = self.bids[bid_price]
            row.append(int(ask_price))
            row.append(int(ask_size))
            row.append(int(bid_price))
            row.append(int(bid_size))

        return row

    def print_order_book(self):
        orderbook = self.get_orderbook_list()
        print(orderbook)

    def validate_orderbook(self, n):
        orderbook_calculated = self.get_orderbook_list()
        orderbook_official = self.orderbook_official.iloc[n].tolist()

        return orderbook_calculated == orderbook_official


    def filter_same_time_data(self):
        # Keep only the last row for each 'Time (sec)' value
        self.metrics_calculated = self.metrics_calculated.drop_duplicates(subset=['Time (sec)'], keep='last').reset_index(drop=True)
        
        
    def calculate_interpolators(self):
        self.interpolators = {}
        
        for column in self.metrics_calculated.columns:
            self.interpolators[column] = interp1d(
                self.metrics_calculated['Time (sec)'],
                self.metrics_calculated[column],
                kind='previous'
            )
