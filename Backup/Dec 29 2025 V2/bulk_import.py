#!/usr/bin/env python
"""
Bulk Import Script - 300+ Binance Futures Pairs
================================================
Run this to import historical data for ALL major coins.

Usage:
    python bulk_import.py --api-key YOUR_COINGLASS_KEY --days 365

Estimated time: ~2-3 hours (rate limited at 30 req/min)
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Comprehensive list of Binance USDT Perpetual Futures (300+)
ALL_SYMBOLS = [
    # Top 20 (already imported - will skip duplicates)
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
    "MATICUSDT", "LTCUSDT", "ATOMUSDT", "UNIUSDT", "XLMUSDT",
    "ETCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    
    # Major Layer 1s
    "SUIUSDT", "SEIUSDT", "TIAUSDT", "INJUSDT", "FTMUSDT",
    "TONUSDT", "TRXUSDT", "ICPUSDT", "HBARUSDT", "ALGOUSDT",
    "FILUSDT", "EGLDUSDT", "FLOWUSDT", "VETUSDT", "NEOUSDT",
    "EOSUSDT", "XTZUSDT", "ZILUSDT", "IOTAUSDT", "QNTUSDT",
    "STXUSDT", "RUNEUSDT", "KASUSDT", "MINAUSDT", "CFXUSDT",
    
    # DeFi
    "AAVEUSDT", "MKRUSDT", "LDOUSDT", "COMPUSDT", "SNXUSDT",
    "CRVUSDT", "DYDXUSDT", "1INCHUSDT", "SUSHIUSDT", "YFIUSDT",
    "GMXUSDT", "PENDLEUSDT", "JUPUSDT", "RAYUSDT", "ORCAUSDT",
    "JABORAUSDT", "ENAUSDT", "ONDOUSDT", "PYTHUSDT", "JTOUSDT",
    
    # Gaming & Metaverse
    "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "ENJUSDT",
    "IMXUSDT", "FLOWUSDT", "ROSEUSDT", "HIGHUSDT", "APEUSDT",
    "GMTUSDT", "MAGICUSDT", "ILVUSDT", "YGGUSDT", "ALICEUSDT",
    
    # AI & Data
    "FETUSDT", "RENDERUSDT", "TAOUSDT", "WLDUSDT", "OCEANUSDT",
    "AGIXUSDT", "ARKMUSDT", "AKTUSDT", "GRTUSDT", "RLCUSDT",
    
    # Memecoins
    "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT",
    "MEMEUSDT", "TURBOUSDT", "BOMEUSDT", "NEIROUSDT", "DOGSUSDT",
    "PNUTUSDT", "ACTUSDT", "TRUMPUSDT", "1000SATSUSDT", "PEOPLEUSDT",
    "1MBABYDOGEUSDT", "LUNCUSDT", "USTCUSDT", "CATUSDT", "MOGUSDT",
    
    # Infrastructure & Scaling
    "ARBUSDT", "OPUSDT", "STRKUSDT", "ZKUSDT", "SCROLLUSDT",
    "MANTAUSDT", "BLASTUSDT", "LINEAUSDT", "BASEDUSDT", "MODUSDT",
    "METISUSDT", "CELOUSDT", "BOBAUSDT", "LOKAUSDT", "FUSIONUSDT",
    
    # Exchange Tokens
    "BNBUSDT", "CAKEUSDT", "FTTUSDT", "CROCUSDT", "HTUSDT",
    "OKBUSDT", "MXUSDT", "GTUSDT", "BNXUSDT", "SFDUSDT",
    
    # Privacy & Storage
    "XMRUSDT", "ZECUSDT", "DASHUSDT", "SCRTUSDT", "ROSAUSDT",
    "STORJUSDT", "ARUSDT", "BLZUSDT", "SIACUSDT", "BTFUSDT",
    
    # Oracle & Bridge
    "LINKUSDT", "BANDUSDT", "APIUSDT", "DIAUSDT", "UMAUSDT",
    "AXLUSDT", "WORMHOLEUSDT", "ZROUSDT", "STGUSDT", "ACXUSDT",
    
    # More Popular Alts
    "BCHUSDT", "XMRUSDT", "ZECUSDT", "DASHUSDT", "ZENUSDT",
    "BATUSDT", "ZRXUSDT", "LRCUSDT", "KSMUSDT", "CHZUSDT",
    "HOTUSDT", "ONEUSDT", "RVNUSDT", "CTSIUSDT", "CELRUSDT",
    "SKLUSDT", "DENTUSDT", "MTLUSDT", "OGNUSDT", "NKNUSDT",
    "OCEANUSDT", "REQUSDT", "GTCUSDT", "MASKUSDT", "AUDIOUSDT",
    "LITUSDT", "PHAUSDT", "KLAYUSDT", "SFPUSDT", "COTIUSDT",
    
    # Newer & Trending
    "EIGENUSDT", "ZKCUSDT", "WUSDT", "MOVEUSDT", "LAYERUSDT",
    "PLUMEUDT", "BERAUSDT", "INITUSDT", "RESOLVUSDT", "FORMUDST",
    "VIRTUALUSDT", "AI16ZUSDT", "GRIFFAINUSDT", "SWARMSUDT",
    "SPELLUSDT", "FABORICUSDT", "IPUSDT", "PARTIALUSDT", "ANIMEUDT",
    
    # Staking & Liquid Staking
    "LDOUSDT", "RETHUSDT", "CBETHUSDT", "SFRXETHUSDT", "WBETHUSDT",
    "ETHFIUSDT", "SSAUSDT", "RPLUSDT", "ANKRUSDT", "STAKEWISEUDT",
    
    # More Infrastructure
    "ENSUSDT", "LPTUSDT", "LIVEPEERUSDT", "THETAUSDT", "TFUELUSDT",
    "POKTUSDT", "NYMUSDT", "XXUSDT", "HNTUSDT", "IOTXUSDT",
    
    # DeFi continued
    "BALUSDT", "UMAAUSDT", "PERPUSDT", "RDNTUSDT", "GRAILUSDT",
    "VELODROMEUSDT", "AERODROMEUDT", "CAMELOTUSDT", "TRADERJOEUSDT",
    
    # More Gaming
    "PRIMESDT", "BEAMUSDT", "XAIUSDT", "RONINUSDT", "PIXELUSDT",
    "PORTALUSDT", "SAGAUSDT", "ACEUSDT", "GASUSDT", "CYBERUSDT",
    
    # More Memes & Culture
    "WOOUSDT", "BONKUSDT", "MYROUST", "WENDYUST", "BRETTUSDT",
    "TOSHIUSDT", "BASEDUSDT", "DEGENUDT", "HIGHERUSDT", "NORMIEUSDT",
    
    # Real World Assets
    "PAXGUSDT", "XAUTUSDT", "GLMRUSDT", "POLUXSDT", "MPLUSDT",
    "REALUSDT", "ABORAUSDT", "LANDUSDT", "PROPYUSDT", "REALITUSDT",
    
    # Additional popular ones
    "ORDIUSDT", "SATSUSDT", "RATSUSDT", "PIXELUSDT", "VANRYUSDT",
    "ALTUSDT", "JUPUSDT", "WUSDT", "ZETAUSDT", "DYMUSDT",
    "MANTAUSDT", "AIUSDT", "XAIUSDT", "MNTUSDT", "TIASUDT",
    "BEAMXUSDT", "NFPUSDT", "ACEUSDT", "XVSUSDT", "BIOUSDT",
    "CKBUSDT", "MAVUSDT", "WLDUSDT", "ARKUSDT", "AGLDUSDT",
    "RADUSDT", "MDTUSDT", "XVSUSD", "EDUUSDT", "IDUSDT",
    "LEVERUSDT", "AMBUSDT", "GASUSDT", "POWRUSDT", "SLPUSDT",
    "TUSDT", "COMBOUSDT", "MAVUSDT", "PENDELUSDT", "ARKMUSDT",
    "WAXPUSDT", "GLMUSDT", "FIDAUSDT", "FRONTUSDT", "TLMUSDT",
    "OMUSDT", "UMAUSDT", "BADGERUSDT", "PERLUSDT", "SUPERUSDT",
    "HARDUSDT", "FORTHUSDT", "BURGERUSDT", "FISUSDT", "PERPUSDT",
    "ALPACAUSDT", "QUICKUSDT", "FARMUSDT", "REQUSDT", "ERNUSDT",
    "IDEXUSDT", "POLYXUSDT", "PHBUDT", "VIBUSDT", "PROSUSDT",
    "SYSUSDT", "DATAUSDT", "ELFUSDT", "NBSUSDT", "CVXUSDT",
    "KLAUSDT", "QIUSDT", "MIRUSDT", "OXTUSDT", "DEGOUSDT",
]

def main():
    parser = argparse.ArgumentParser(description="Bulk import historical data")
    parser.add_argument("--api-key", required=True, help="Coinglass API key")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--batch-size", type=int, default=50, help="Coins per batch")
    args = parser.parse_args()
    
    # Remove duplicates and clean
    symbols = list(dict.fromkeys([s.upper() for s in ALL_SYMBOLS if s.endswith('USDT')]))
    
    print("=" * 60)
    print(f"BULK IMPORT: {len(symbols)} symbols, {args.days} days")
    print("=" * 60)
    print(f"Estimated time: {len(symbols) * 5 * 2.5 / 60:.0f} minutes")
    print("=" * 60)
    
    from core.historical_data_fetcher import HistoricalDataImporter
    
    importer = HistoricalDataImporter(args.api_key)
    
    # Import in batches
    total_records = 0
    total_symbols = 0
    failed = []
    
    for i in range(0, len(symbols), args.batch_size):
        batch = symbols[i:i + args.batch_size]
        print(f"\n{'='*60}")
        print(f"BATCH {i//args.batch_size + 1}: {batch[0]} - {batch[-1]}")
        print(f"{'='*60}")
        
        stats = importer.import_historical_data(
            symbols=batch,
            lookback_days=args.days,
            interval="4h"
        )
        
        total_records += stats['records_imported']
        total_symbols += stats['symbols_processed']
        failed.extend(stats.get('symbols_failed', []))
    
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Total records: {total_records:,}")
    print(f"Symbols imported: {total_symbols}")
    if failed:
        print(f"Failed: {len(failed)} - {failed[:10]}...")


if __name__ == "__main__":
    main()
