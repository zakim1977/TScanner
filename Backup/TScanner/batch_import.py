#!/usr/bin/env python
"""
Batch Import Script - Import in chunks of 40 coins
===================================================
Run batches when convenient - safe to re-run (won't duplicate)

Usage:
    python batch_import.py --api-key YOUR_KEY --batch 1
    python batch_import.py --api-key YOUR_KEY --batch 2
    python batch_import.py --api-key YOUR_KEY --batch all

Each batch: ~40 coins, ~25-30 minutes
Total: 7 batches = ~280 coins = ~3 hours
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH DEFINITIONS - 40 coins each
# ═══════════════════════════════════════════════════════════════════════════════

BATCHES = {
    1: {
        "name": "DeFi + AI Tokens",
        "symbols": [
            "AAVEUSDT", "MKRUSDT", "LDOUSDT", "COMPUSDT", "SNXUSDT", 
            "CRVUSDT", "DYDXUSDT", "SUSHIUSDT", "PENDLEUSDT", "JUPUSDT",
            "ONDOUSDT", "PYTHUSDT", "FETUSDT", "RENDERUSDT", "TAOUSDT", 
            "WLDUSDT", "GRTUSDT", "ARKMUSDT", "INJUSDT", "RUNEUSDT",
            "SUIUSDT", "SEIUSDT", "TIAUSDT", "FTMUSDT", "TONUSDT", 
            "ICPUSDT", "HBARUSDT", "ALGOUSDT", "FILUSDT", "STXUSDT",
            "EIGENUSDT", "STRKUSDT", "ZKUSDT", "MOVEUSDT", "LAYERUSDT", 
            "BERAUSDT", "INITUSDT", "VIRTUALUSDT", "AXLUSDT", "ENSUSDT"
        ]
    },
    2: {
        "name": "Memecoins + Gaming",
        "symbols": [
            "SHIBUSDT", "PEPEUSDT", "WIFUSDT", "BONKUSDT", "FLOKIUSDT", 
            "MEMEUSDT", "TURBOUSDT", "BOMEUSDT", "NEIROUSDT", "DOGSUSDT",
            "PNUTUSDT", "ACTUSDT", "TRUMPUSDT", "PEOPLEUSDT", "LUNCUSDT", 
            "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "ENJUSDT",
            "IMXUSDT", "APEUSDT", "GMTUSDT", "MAGICUSDT", "YGGUSDT", 
            "ALICEUSDT", "PORTALUSDT", "ACEUSDT", "PIXELUSDT", "XAIUSDT",
            "HIGHUSDT", "BEAMXUSDT", "PRIMEUSDT", "RONINUSDT", "GASUSDT", 
            "ILVUSDT", "FLOWUSDT", "ROSEUSDT", "HOTUSDT", "CHZUSDT"
        ]
    },
    3: {
        "name": "Infrastructure + Legacy",
        "symbols": [
            "BCHUSDT", "ZECUSDT", "DASHUSDT", "ZENUSDT", "BATUSDT", 
            "ZRXUSDT", "LRCUSDT", "MASKUSDT", "AUDIOUSDT", "LITUSDT",
            "COTIUSDT", "CELOUSDT", "METISUSDT", "ANKRUSDT", "LPTUSDT", 
            "THETAUSDT", "IOTXUSDT", "BALUSDT", "RDNTUSDT", "QNTUSDT",
            "NEOUSDT", "XTZUSDT", "ZILUSDT", "CFXUSDT", "EGLDUSDT", 
            "VETUSDT", "KLAYUSDT", "SFPUSDT", "ORDIUSDT", "SATSUSDT",
            "DYMUSDT", "MANTAUSDT", "ZETAUSDT", "ALTUSDT", "MNTUSDT", 
            "NFPUSDT", "MAVUSDT", "WAXPUSDT", "GLMRUSDT", "CKBUSDT"
        ]
    },
    4: {
        "name": "Layer 2 + Scaling",
        "symbols": [
            "ARBUSDT", "OPUSDT", "MATICUSDT", "SCROLLUSDT", "BLASTUSDT",
            "LINEAUSDT", "MODUSDT", "BOBAUSDT", "LOKAUSDT", "FUSIONUSDT",
            "OMGUSDT", "CELRUSDT", "SKLUSDT", "CTSIUSDT", "BOSONUSDT",
            "REQUSDT", "GTCUSDT", "RADUSDT", "MDTUSDT", "LEVERUSDT",
            "AMBUSDT", "POWRUSDT", "SLPUSDT", "TUSDT", "COMBOUSDT",
            "FIDAUSDT", "FRONTUSDT", "TLMUSDT", "OMUSDT", "UMAUSDT",
            "BADGERUSDT", "PERPUSDT", "SUPERUSDT", "FORTHUSDT", "FISUSDT",
            "ALPACAUSDT", "QUICKUSDT", "FARMUSDT", "ERNUSDT", "IDEXUSDT"
        ]
    },
    5: {
        "name": "Exchange + Oracle Tokens",
        "symbols": [
            "CAKEUSDT", "BNXUSDT", "WOOUSDT", "DEXEUSDT", "JABORAUSDT",
            "BANDUSDT", "API3USDT", "DIAUSDT", "PYRUSDT", "WORMHOLEUSDT",
            "ZROUSDT", "STGUSDT", "SYNUSDT", "CELAUSDT", "HOPUSDT",
            "ACXUSDT", "LIFIUSDT", "ROUTEUSDT", "SWFTUSDT", "MULTIUSDT",
            "PAXGUSDT", "XAUTUSDT", "GLMRUSDT", "POLYXUSDT", "PROPYUSDT",
            "STORJUSDT", "ARUSDT", "BLZUSDT", "FILAUSDT", "HIVEUSDT",
            "SCUSDT", "XEMUSDT", "STEEMUDST", "ABORAUSDT", "LANDUSDT",
            "REALUSDT", "PROSUSDT", "SYSUSDT", "DATAUSDT", "ELFUSDT"
        ]
    },
    6: {
        "name": "New Listings + Trending",
        "symbols": [
            "BIOUSDT", "USUALUSDT", "VANRYUSDT", "SAGAUSDT", "WUSDT",
            "JTOUSDT", "ETHFIUSDT", "ENAUSDT", "BBUSDT", "NOTUSDT",
            "IOUSDT", "REZUSDT", "LISTAUSDT", "UXLINKUSDT", "SCRUSDT",
            "CATIUSDT", "HMSTRUSDT", "DEGOUSDT", "UXDUSDT", "DEFIUSDT",
            "BLURUSDT", "EDUUSDT", "IDUSDT", "CYBERUSDT", "ARKUSDT",
            "AGLDUSDT", "TRUUSDT", "ONGUSDT", "KEYUSDT", "VITEUSDT",
            "OAXUSDT", "DABORAUSDT", "NEXOUSDT", "FUNUSDT", "PHAUSDT",
            "AEROUSDT", "VELODROMEUSDT", "EXTRAUSDT", "GMXUSDT", "GRAILUSDT"
        ]
    },
    7: {
        "name": "Additional Coverage",
        "symbols": [
            "TRXUSDT", "EOSUSDT", "IOSTUSDT", "ONTUSDT", "WAVESUSDT",
            "KAVAUSDT", "KSMUSDT", "MINAUSDT", "AUCTIONUSDT", "PHBUSDT",
            "VIBUSDT", "NMRUSDT", "OXTUSDT", "QIUSDT", "MIRUSDT",
            "CVXUSDT", "KLAUSDT", "NBSUSDT", "SUNUSDT", "JSTUSDT",
            "WINUSDT", "BTTUSDT", "DENTUSDT", "MTLUSDT", "OGNUSDT",
            "NKNUSDT", "OCEANUSDT", "RVNUSDT", "ONEUSDT", "HNTUSDT",
            "RSRUSDT", "SANTOSUSDT", "LAZIOUSDT", "ALPINEUSDT", "ASRUSDT",
            "ATMUSDT", "BARUSDT", "CITYUSDT", "PORTOUSDT", "ACMUSDT"
        ]
    }
}


def get_imported_symbols(db_path: str = "data/whale_history.db") -> set:
    """Check which symbols already have sufficient data"""
    import sqlite3
    
    if not os.path.exists(db_path):
        return set()
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Consider "imported" if has more than 100 records
        cursor.execute("""
            SELECT symbol FROM whale_snapshots 
            GROUP BY symbol HAVING COUNT(*) > 100
        """)
        symbols = {row[0] for row in cursor.fetchall()}
        conn.close()
        return symbols
    except:
        return set()


def run_batch(api_key: str, batch_num: int, days: int = 365, skip_imported: bool = True):
    """Run a single batch import"""
    
    if batch_num not in BATCHES:
        print(f"❌ Invalid batch number: {batch_num}")
        print(f"   Available batches: 1-{len(BATCHES)}")
        return
    
    batch = BATCHES[batch_num]
    symbols = batch["symbols"]
    
    # Check what's already imported
    if skip_imported:
        imported = get_imported_symbols()
        remaining = [s for s in symbols if s not in imported]
        skipped = len(symbols) - len(remaining)
        
        if skipped > 0:
            print(f"⏭️  Skipping {skipped} already-imported symbols")
        
        if not remaining:
            print(f"✅ Batch {batch_num} already fully imported!")
            return
        
        symbols = remaining
    
    print("=" * 60)
    print(f"BATCH {batch_num}: {batch['name']}")
    print(f"Symbols: {len(symbols)} | Days: {days}")
    print(f"Estimated time: ~{len(symbols) * 4 * 2.5 / 60:.0f} minutes")
    print("=" * 60)
    
    from core.historical_data_fetcher import HistoricalDataImporter
    
    importer = HistoricalDataImporter(api_key)
    stats = importer.import_historical_data(
        symbols=symbols,
        lookback_days=days,
        interval="4h"
    )
    
    print("\n" + "=" * 60)
    print(f"BATCH {batch_num} COMPLETE")
    print(f"Records imported: {stats['records_imported']:,}")
    print(f"Symbols processed: {stats['symbols_processed']}")
    if stats.get('symbols_failed'):
        print(f"Failed: {stats['symbols_failed']}")
    print("=" * 60)


def show_status():
    """Show import status for all batches"""
    imported = get_imported_symbols()
    
    print("\n" + "=" * 60)
    print("IMPORT STATUS")
    print("=" * 60)
    
    total_symbols = 0
    total_imported = 0
    
    for batch_num, batch in BATCHES.items():
        symbols = batch["symbols"]
        done = sum(1 for s in symbols if s in imported)
        total_symbols += len(symbols)
        total_imported += done
        
        status = "✅" if done == len(symbols) else "🔄" if done > 0 else "⬚"
        print(f"{status} Batch {batch_num}: {batch['name'][:25]:<25} {done}/{len(symbols)}")
    
    print("-" * 60)
    print(f"Total: {total_imported}/{total_symbols} symbols imported")
    print(f"Already in DB: {len(imported)} symbols with >100 records")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Import historical data in batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_import.py --status                    # Check what's imported
  python batch_import.py --api-key KEY --batch 1    # Run batch 1
  python batch_import.py --api-key KEY --batch 2    # Run batch 2
  python batch_import.py --api-key KEY --batch all  # Run all batches
        """
    )
    parser.add_argument("--api-key", help="Coinglass API key")
    parser.add_argument("--batch", help="Batch number (1-7) or 'all'")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--status", action="store_true", help="Show import status")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip already-imported symbols")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    if not args.api_key or not args.batch:
        parser.print_help()
        print("\n❌ Required: --api-key and --batch")
        return
    
    if args.batch.lower() == "all":
        print("🚀 Running ALL batches - this will take ~3 hours")
        for batch_num in BATCHES:
            run_batch(args.api_key, batch_num, args.days, not args.no_skip)
    else:
        try:
            batch_num = int(args.batch)
            run_batch(args.api_key, batch_num, args.days, not args.no_skip)
        except ValueError:
            print(f"❌ Invalid batch: {args.batch}")
            print(f"   Use 1-{len(BATCHES)} or 'all'")


if __name__ == "__main__":
    main()
