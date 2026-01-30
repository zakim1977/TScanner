"""
Coinglass Real Liquidation Heatmap Integration
===============================================
Fetches REAL liquidation clusters from Coinglass API for accurate TP placement.

API Endpoints:
- /api/futures/liquidation/heatmap/model1 - Liquidation heatmap by pair
- /api/futures/liquidation/map - Liquidation map with prices and volumes

Replaces calculated formulas with REAL data!
"""

import os
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class CoinglassLiquidation:
    """
    Fetches real liquidation data from Coinglass API.
    
    Returns actual liquidation clusters where positions will be liquidated,
    with $ volume at each level.
    """
    
    BASE_URL = "https://open-api-v4.coinglass.com"
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Coinglass API key.
        
        Args:
            api_key: Coinglass API key. If None, tries to get from:
                     1. Environment variable COINGLASS_API_KEY
                     2. Streamlit session state
        """
        self.api_key = api_key
        
        if not self.api_key:
            # Try environment variable
            self.api_key = os.environ.get('COINGLASS_API_KEY', '')
        
        if not self.api_key:
            # Try to get from Streamlit session state
            try:
                import streamlit as st
                self.api_key = st.session_state.get('settings', {}).get('coinglass_api_key', '')
            except:
                pass
        
        self.headers = {
            "Accept": "application/json",
            "CG-API-KEY": self.api_key
        } if self.api_key else {}
        
        self._rate_limit_delay = 2.5  # Hobbyist tier: 30 req/min
        self._cache = {}
        self._cache_ttl = 60  # Cache for 60 seconds
        self._last_request = 0
    
    def is_available(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)
    
    def _request(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make rate-limited API request with caching."""
        if not self.api_key:
            return None
        
        # Create cache key
        cache_key = f"{endpoint}:{str(params)}"
        
        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
        
        # Rate limit
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=15)
            self._last_request = time.time()
            
            if response.status_code == 200:
                data = response.json()
                code = data.get('code')
                
                if code == "0" or code == 0:
                    # Cache successful response
                    self._cache[cache_key] = (time.time(), data)
                    return data
                else:
                    print(f"[COINGLASS_LIQ] API Error: {data.get('msg', 'Unknown')}")
                    return None
                    
            elif response.status_code == 429:
                print("[COINGLASS_LIQ] Rate limited - waiting 60s")
                time.sleep(60)
                return self._request(endpoint, params)
            elif response.status_code == 401:
                print("[COINGLASS_LIQ] Invalid API key")
                return None
            else:
                print(f"[COINGLASS_LIQ] HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[COINGLASS_LIQ] Request error: {e}")
            return None
    
    def get_liquidation_heatmap(
        self, 
        symbol: str, 
        exchange: str = "Binance",
        range_type: str = "12h"
    ) -> Optional[Dict]:
        """
        Get liquidation heatmap for a trading pair.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            exchange: Exchange name (default: Binance)
            range_type: Time range - '12h', '24h', '3d', '7d', '30d', '90d', '180d', '1y'
        
        Returns:
            Dict with liquidation data:
            {
                'success': True,
                'current_price': 102500,
                'levels': [
                    {'price': 103500, 'volume': 15793707, 'leverage': 25, 'side': 'SHORT'},
                    {'price': 101500, 'volume': 8234521, 'leverage': 50, 'side': 'LONG'},
                    ...
                ],
                'above_price': [...],  # SHORT liquidations (targets for LONG trades)
                'below_price': [...],  # LONG liquidations (targets for SHORT trades)
                'total_above': 45000000,
                'total_below': 38000000
            }
        """
        # Ensure symbol format
        symbol_clean = symbol.upper()
        if not symbol_clean.endswith('USDT'):
            symbol_clean += 'USDT'
        
        params = {
            "exchange": exchange,
            "symbol": symbol_clean,
            "range": range_type
        }
        
        result = self._request("/api/futures/liquidation/heatmap/model1", params)
        
        if not result or not result.get('data'):
            return None
        
        return self._parse_heatmap_data(result.get('data', {}), symbol_clean)
    
    def get_liquidation_map(
        self, 
        symbol: str, 
        exchange: str = "Binance"
    ) -> Optional[Dict]:
        """
        Get liquidation map with price levels and volumes.
        
        This endpoint gives more granular data with actual $ amounts.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            exchange: Exchange name
        
        Returns:
            Dict with liquidation levels and volumes
        """
        symbol_clean = symbol.upper()
        if not symbol_clean.endswith('USDT'):
            symbol_clean += 'USDT'
        
        params = {
            "exchange": exchange,
            "symbol": symbol_clean
        }
        
        result = self._request("/api/futures/liquidation/map", params)
        
        if not result or not result.get('data'):
            return None
        
        return self._parse_map_data(result.get('data', {}), symbol_clean)
    
    def _parse_heatmap_data(self, data: Dict, symbol: str) -> Dict:
        """Parse heatmap API response into usable format."""
        try:
            levels = []
            above_price = []
            below_price = []
            
            # Get current price from data if available
            current_price = data.get('price', 0)
            
            # Parse the heatmap data structure
            # Format: { "price_level": [[price, volume, leverage, null], ...], ... }
            heatmap_data = data.get('data', data)
            
            if isinstance(heatmap_data, dict):
                for price_key, price_data in heatmap_data.items():
                    try:
                        if isinstance(price_data, list):
                            for item in price_data:
                                if isinstance(item, list) and len(item) >= 3:
                                    price = float(item[0])
                                    volume = float(item[1]) if item[1] else 0
                                    leverage = int(item[2]) if item[2] else 0
                                    
                                    if price <= 0 or volume <= 0:
                                        continue
                                    
                                    # Determine side based on price relative to current
                                    if current_price > 0:
                                        side = 'SHORT' if price > current_price else 'LONG'
                                    else:
                                        # Infer from leverage position
                                        side = 'UNKNOWN'
                                    
                                    level = {
                                        'price': price,
                                        'volume': volume,
                                        'leverage': leverage,
                                        'side': side
                                    }
                                    
                                    levels.append(level)
                                    
                                    if side == 'SHORT':
                                        above_price.append(level)
                                    elif side == 'LONG':
                                        below_price.append(level)
                    except (ValueError, TypeError):
                        continue
            
            # Sort by volume (highest first)
            levels.sort(key=lambda x: x['volume'], reverse=True)
            above_price.sort(key=lambda x: x['price'])  # Closest first
            below_price.sort(key=lambda x: x['price'], reverse=True)  # Closest first
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'levels': levels,
                'above_price': above_price[:20],  # Top 20 levels
                'below_price': below_price[:20],
                'total_above': sum(l['volume'] for l in above_price),
                'total_below': sum(l['volume'] for l in below_price)
            }
            
        except Exception as e:
            print(f"[COINGLASS_LIQ] Parse error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_map_data(self, data: Dict, symbol: str) -> Dict:
        """Parse liquidation map API response."""
        try:
            # Similar parsing logic for map endpoint
            return self._parse_heatmap_data(data, symbol)
        except Exception as e:
            print(f"[COINGLASS_LIQ] Map parse error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_tp_levels_from_liquidation(
        self,
        symbol: str,
        current_price: float,
        direction: str,
        num_levels: int = 3,
        min_volume: float = 1000000  # Minimum $1M volume
    ) -> List[Dict]:
        """
        Get TP levels based on real liquidation clusters.
        
        For LONG trades: Target SHORT liquidation levels (above price)
        For SHORT trades: Target LONG liquidation levels (below price)
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            direction: 'LONG' or 'SHORT'
            num_levels: Number of TP levels to return
            min_volume: Minimum liquidation volume to consider
        
        Returns:
            List of TP levels:
            [
                {'price': 103500, 'volume': 15793707, 'leverage': 25, 'reason': 'Liquidation Cluster ($15.8M)'},
                ...
            ]
        """
        # Try heatmap first, then map
        liq_data = self.get_liquidation_heatmap(symbol)
        
        if not liq_data or not liq_data.get('success'):
            liq_data = self.get_liquidation_map(symbol)
        
        if not liq_data or not liq_data.get('success'):
            return []
        
        # Select levels based on direction
        if direction.upper() == 'LONG':
            # Target SHORT liquidations (above price) for LONG trades
            candidates = liq_data.get('above_price', [])
        else:
            # Target LONG liquidations (below price) for SHORT trades
            candidates = liq_data.get('below_price', [])
        
        # Filter by minimum volume and sort by proximity to current price
        filtered = []
        for level in candidates:
            if level['volume'] >= min_volume:
                # Calculate distance from current price
                distance_pct = abs(level['price'] - current_price) / current_price * 100
                
                # Skip if too close (< 0.5%) or too far (> 15%)
                if 0.5 <= distance_pct <= 15:
                    level['distance_pct'] = distance_pct
                    level['reason'] = f"Liquidation Cluster (${level['volume']/1e6:.1f}M)"
                    filtered.append(level)
        
        # Sort by distance (closest first)
        if direction.upper() == 'LONG':
            filtered.sort(key=lambda x: x['price'])
        else:
            filtered.sort(key=lambda x: x['price'], reverse=True)
        
        return filtered[:num_levels]


# Singleton instance
_coinglass_liq = None

def get_coinglass_liquidation(api_key: str = None) -> CoinglassLiquidation:
    """Get or create singleton instance."""
    global _coinglass_liq
    
    if _coinglass_liq is None or (api_key and api_key != _coinglass_liq.api_key):
        _coinglass_liq = CoinglassLiquidation(api_key)
    
    return _coinglass_liq


def get_real_liquidation_levels(
    symbol: str,
    current_price: float,
    direction: str = None,
    api_key: str = None
) -> Dict:
    """
    Convenience function to get real liquidation data.
    
    Args:
        symbol: Trading pair
        current_price: Current price
        direction: Optional - 'LONG' or 'SHORT' to get targeted TPs
        api_key: Optional Coinglass API key
    
    Returns:
        Dict with:
        - 'success': bool
        - 'source': 'coinglass' or 'calculated'
        - 'levels': List of liquidation levels
        - 'tps': List of TP targets (if direction specified)
        - 'above_price': Levels above current price
        - 'below_price': Levels below current price
    """
    cg = get_coinglass_liquidation(api_key)
    
    if not cg.is_available():
        # Fall back to calculated
        return _calculate_fallback_levels(symbol, current_price, direction)
    
    # Try to get real data
    liq_data = cg.get_liquidation_heatmap(symbol)
    
    if not liq_data or not liq_data.get('success'):
        return _calculate_fallback_levels(symbol, current_price, direction)
    
    result = {
        'success': True,
        'source': 'coinglass',
        'symbol': symbol,
        'current_price': current_price,
        'levels': liq_data.get('levels', []),
        'above_price': liq_data.get('above_price', []),
        'below_price': liq_data.get('below_price', []),
        'total_above': liq_data.get('total_above', 0),
        'total_below': liq_data.get('total_below', 0)
    }
    
    # Get TP levels if direction specified
    if direction:
        result['tps'] = cg.get_tp_levels_from_liquidation(
            symbol, current_price, direction, num_levels=3
        )
    
    return result


def _calculate_fallback_levels(symbol: str, current_price: float, direction: str = None) -> Dict:
    """
    Calculate estimated liquidation levels when API not available.
    
    Uses standard leverage formulas:
    - 100x: ±1% from price
    - 50x: ±2% from price
    - 25x: ±4% from price
    """
    levels = [
        # Above price (SHORT liquidations)
        {'price': current_price * 1.01, 'volume': 0, 'leverage': 100, 'side': 'SHORT', 'reason': 'Short Liq (100x) - Estimated'},
        {'price': current_price * 1.02, 'volume': 0, 'leverage': 50, 'side': 'SHORT', 'reason': 'Short Liq (50x) - Estimated'},
        {'price': current_price * 1.04, 'volume': 0, 'leverage': 25, 'side': 'SHORT', 'reason': 'Short Liq (25x) - Estimated'},
        # Below price (LONG liquidations)
        {'price': current_price * 0.99, 'volume': 0, 'leverage': 100, 'side': 'LONG', 'reason': 'Long Liq (100x) - Estimated'},
        {'price': current_price * 0.98, 'volume': 0, 'leverage': 50, 'side': 'LONG', 'reason': 'Long Liq (50x) - Estimated'},
        {'price': current_price * 0.96, 'volume': 0, 'leverage': 25, 'side': 'LONG', 'reason': 'Long Liq (25x) - Estimated'},
    ]
    
    above = [l for l in levels if l['side'] == 'SHORT']
    below = [l for l in levels if l['side'] == 'LONG']
    
    result = {
        'success': True,
        'source': 'calculated',
        'symbol': symbol,
        'current_price': current_price,
        'levels': levels,
        'above_price': above,
        'below_price': below,
        'total_above': 0,
        'total_below': 0
    }
    
    if direction:
        if direction.upper() == 'LONG':
            result['tps'] = above[:3]
        else:
            result['tps'] = below[:3]
    
    return result


# Test
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("COINGLASS LIQUIDATION HEATMAP TEST")
    print("=" * 60)
    
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not api_key:
        print("Usage: python coinglass_liquidation.py YOUR_API_KEY")
        print("\nTesting with calculated fallback...")
        result = get_real_liquidation_levels('BTCUSDT', 100000, 'LONG')
    else:
        print(f"Testing with API key: {api_key[:8]}...")
        result = get_real_liquidation_levels('BTCUSDT', 100000, 'LONG', api_key)
    
    print(f"\nSource: {result.get('source')}")
    print(f"Success: {result.get('success')}")
    
    if result.get('tps'):
        print(f"\nTP Levels for LONG trade:")
        for tp in result['tps']:
            print(f"  ${tp['price']:,.2f} - {tp.get('reason', 'N/A')}")
    
    print(f"\nLevels above price: {len(result.get('above_price', []))}")
    print(f"Levels below price: {len(result.get('below_price', []))}")
    print(f"Total $ above: ${result.get('total_above', 0):,.0f}")
    print(f"Total $ below: ${result.get('total_below', 0):,.0f}")
