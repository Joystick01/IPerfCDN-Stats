#!/usr/bin/env python3
"""
DynamoDB JSON to Parquet Converter

Konvertiert DynamoDB JSON Dateien in Parquet Format mit spezifischem Schema.
Verwendet PySpark für effiziente Verarbeitung großer Dateien.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Union

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, 
    TimestampType, ArrayType
)
from pyspark.sql.functions import col, from_json, lit


def parse_dynamodb_value(value: Dict[str, Any]) -> Any:
    """
    Konvertiert DynamoDB Typen in normale Python-Typen.
    
    Args:
        value: DynamoDB Wert mit Typ-Annotation
        
    Returns:
        Konvertierter Python-Wert
    """
    if 'S' in value:  # String
        return value['S']
    elif 'N' in value:  # Number
        num_str = value['N']
        # Versuche Integer, falls das fehlschlägt, verwende Float
        try:
            if '.' in num_str:
                return float(num_str)
            else:
                return int(num_str)
        except ValueError:
            return float(num_str)
    elif 'L' in value:  # List
        return [parse_dynamodb_value(item) for item in value['L']]
    elif 'M' in value:  # Map
        return {k: parse_dynamodb_value(v) for k, v in value['M'].items()}
    elif 'BOOL' in value:  # Boolean
        return value['BOOL']
    elif 'NULL' in value:  # Null
        return None
    else:
        # Fallback: Rückgabe des Werts falls unbekannter Typ
        return value


def convert_dynamodb_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Konvertiert ein DynamoDB Item in ein normales Python Dictionary.
    
    Args:
        item: DynamoDB Item
        
    Returns:
        Konvertiertes Dictionary
    """
    result = {}
    
    if 'Item' in item:
        item_data = item['Item']
    else:
        item_data = item
    
    for key, value in item_data.items():
        result[key] = parse_dynamodb_value(value)
    
    return result


def ensure_schema_compliance(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stellt sicher, dass das Item dem erwarteten Schema entspricht.
    
    Args:
        item: Konvertiertes DynamoDB Item
        
    Returns:
        Schema-konformes Item
    """
    # Setze Default-Werte für fehlende Felder
    defaults = {
        'id': None,
        'utctime': None,
        'bitpattern': None,
        'src': None,
        'psize': None,
        'dst': None,
        'tos': None,
        'tests': None,
        'region': None,
        'hubs': []
    }
    
    # Fülle fehlende Felder mit Defaults
    for key, default_value in defaults.items():
        if key not in item:
            item[key] = default_value
    
    # Stelle sicher, dass hubs eine Liste ist
    if item['hubs'] is None:
        item['hubs'] = []
    elif not isinstance(item['hubs'], list):
        item['hubs'] = []
    
    # Bereinige hub-Einträge
    cleaned_hubs = []
    for hub in item['hubs']:
        if isinstance(hub, dict):
            cleaned_hub = {}
            # Alle erwarteten Hub-Felder mit Defaults
            hub_defaults = {
                'Loss%': 0.0,
                'Rcv': 0,
                'Jint': 0.0,
                'count': 0,
                'Jttr': 0.0,
                'Javg': 0.0,
                'Drop': 0,
                'Last': 0.0,
                'Avg': 0.0,
                'StDev': 0.0,
                'Wrst': 0.0,
                'Gmean': 0.0,
                'host': None,
                'Snt': 0,
                'Jmax': 0.0,
                'Best': 0.0,
                'ASN': None
            }
            
            for key, default_value in hub_defaults.items():
                if key in hub:
                    # Konvertiere zu korrektem Typ
                    value = hub[key]
                    if key in ['Loss%', 'Jint', 'Jttr', 'Javg', 'Last', 'Avg', 'StDev', 'Wrst', 'Gmean', 'Jmax', 'Best']:
                        cleaned_hub[key] = float(value) if value is not None else 0.0
                    elif key in ['Rcv', 'count', 'Drop', 'Snt']:
                        cleaned_hub[key] = int(value) if value is not None else 0
                    else:  # String-Felder
                        cleaned_hub[key] = str(value) if value is not None else None
                else:
                    cleaned_hub[key] = default_value
            
            cleaned_hubs.append(cleaned_hub)
    
    item['hubs'] = cleaned_hubs
    
    # Typ-Konvertierungen für andere Felder
    if item['psize'] is not None:
        try:
            item['psize'] = int(item['psize'])
        except (ValueError, TypeError):
            item['psize'] = None
    
    if item['tos'] is not None:
        try:
            item['tos'] = int(item['tos'])
        except (ValueError, TypeError):
            item['tos'] = None
    
    if item['tests'] is not None:
        try:
            item['tests'] = int(item['tests'])
        except (ValueError, TypeError):
            item['tests'] = None
    
    return item


def apply_schema_transformations(df, spark: SparkSession):
    """
    Wendet Schema-Transformationen auf ein bestehendes DataFrame an.
    
    Args:
        df: Bestehendes DataFrame
        spark: SparkSession
        
    Returns:
        Transformiertes DataFrame
    """
    from pyspark.sql.functions import col, when, lit
    from pyspark.sql.types import TimestampType, IntegerType, DoubleType, StringType
    
    # Basis-Transformationen
    transformed_df = df.select(
        col("id").cast(StringType()).alias("id"),
        col("utctime").cast(TimestampType()).alias("utctime"),
        col("bitpattern").cast(StringType()).alias("bitpattern"),
        col("src").cast(StringType()).alias("src"),
        col("psize").cast(IntegerType()).alias("psize"),
        col("dst").cast(StringType()).alias("dst"),
        col("tos").cast(IntegerType()).alias("tos"),
        col("tests").cast(IntegerType()).alias("tests"),
        col("region").cast(StringType()).alias("region"),
        col("hubs")  # Hubs bleiben wie sie sind, da sie schon richtig strukturiert sind
    )
    
    return transformed_df


def create_schema() -> StructType:
    """
    Erstellt das Parquet Schema basierend auf den Anforderungen.
    
    Returns:
        PySpark StructType Schema
    """
    hub_schema = StructType([
        StructField("Loss%", DoubleType(), True),
        StructField("Rcv", IntegerType(), True),
        StructField("Jint", DoubleType(), True),
        StructField("count", IntegerType(), True),
        StructField("Jttr", DoubleType(), True),
        StructField("Javg", DoubleType(), True),
        StructField("Drop", IntegerType(), True),
        StructField("Last", DoubleType(), True),
        StructField("Avg", DoubleType(), True),
        StructField("StDev", DoubleType(), True),
        StructField("Wrst", DoubleType(), True),
        StructField("Gmean", DoubleType(), True),
        StructField("host", StringType(), True),
        StructField("Snt", IntegerType(), True),
        StructField("Jmax", DoubleType(), True),
        StructField("Best", DoubleType(), True),
        StructField("ASN", StringType(), True)
    ])
    
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("utctime", TimestampType(), True),
        StructField("bitpattern", StringType(), True),
        StructField("src", StringType(), True),
        StructField("psize", IntegerType(), True),
        StructField("dst", StringType(), True),
        StructField("tos", IntegerType(), True),
        StructField("tests", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("hubs", ArrayType(hub_schema), True)
    ])
    
    return schema


def parse_multiple_json_objects(content: str) -> List[Dict[str, Any]]:
    """
    Parst mehrere JSON-Objekte, die direkt hintereinander stehen.
    
    Args:
        content: String mit JSON-Objekten
        
    Returns:
        Liste von geparsten JSON-Objekten
    """
    items = []
    decoder = json.JSONDecoder()
    idx = 0
    
    while idx < len(content):
        content = content[idx:].lstrip()
        if not content:
            break
            
        try:
            obj, end_idx = decoder.raw_decode(content)
            items.append(obj)
            idx += end_idx
        except json.JSONDecodeError as e:
            print(f"Fehler beim Parsen ab Position {idx}: {e}")
            # Versuche zum nächsten '{' zu springen
            next_brace = content.find('{', 1)
            if next_brace == -1:
                break
            idx += next_brace
            continue
    
    return items


def process_file(spark: SparkSession, input_file: str, output_file: str):
    """
    Verarbeitet die DynamoDB JSON Datei und konvertiert sie in Parquet.
    
    Args:
        spark: SparkSession
        input_file: Pfad zur Eingabedatei
        output_file: Pfad zur Ausgabedatei
    """
    print(f"Lese DynamoDB JSON Datei: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        sys.exit(1)
    
    print("Versuche verschiedene JSON-Formate...")
    items = []
    
    # Versuch 1: Einzelnes JSON-Objekt
    try:
        single_obj = json.loads(content)
        items = [single_obj]
        print("✓ Erfolgreich als einzelnes JSON-Objekt geparst")
    except json.JSONDecodeError:
        print("✗ Nicht als einzelnes JSON-Objekt parsbar")
        
        # Versuch 2: JSONL Format (eine JSON-Struktur pro Zeile)
        try:
            items = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"✗ Fehler in Zeile {line_num}: {e}")
                        break
            if items:
                print(f"✓ Erfolgreich als JSONL geparst ({len(items)} Zeilen)")
            else:
                raise ValueError("Keine gültigen JSON-Zeilen gefunden")
        except (json.JSONDecodeError, ValueError):
            print("✗ Nicht als JSONL parsbar")
            
            # Versuch 3: Multiple JSON-Objekte direkt hintereinander
            try:
                items = parse_multiple_json_objects(content)
                if items:
                    print(f"✓ Erfolgreich als multiple JSON-Objekte geparst ({len(items)} Objekte)")
                else:
                    raise ValueError("Keine JSON-Objekte gefunden")
            except Exception as e:
                print(f"✗ Fehler beim Parsen multipler JSON-Objekte: {e}")
                print("\nKonnte die JSON-Datei in keinem unterstützten Format parsen.")
                print("Unterstützte Formate:")
                print("1. Einzelnes JSON-Objekt")
                print("2. JSONL (ein JSON-Objekt pro Zeile)")
                print("3. Multiple JSON-Objekte direkt hintereinander")
                sys.exit(1)
    
    print(f"Gefunden: {len(items)} Datensätze")
    
    # Konvertiere DynamoDB Items
    converted_items = []
    for item in items:
        converted_item = convert_dynamodb_item(item)
        
        # Konvertiere UTC-Zeit String in Timestamp
        if 'utctime' in converted_item and converted_item['utctime']:
            try:
                # Entferne das '+00:00' und parse das Datum
                time_str = converted_item['utctime'].replace('+00:00', '')
                converted_item['utctime'] = datetime.fromisoformat(time_str)
            except (ValueError, AttributeError):
                converted_item['utctime'] = None
        
        # Stelle sicher, dass alle erwarteten Felder vorhanden sind
        converted_item = ensure_schema_compliance(converted_item)
        
        converted_items.append(converted_item)
    
    # Erstelle das Schema
    schema = create_schema()
    
    # Erstelle DataFrame mit explizitem Schema
    print("Erstelle Spark DataFrame mit explizitem Schema...")
    try:
        transformed_df = spark.createDataFrame(converted_items, schema)
    except Exception as e:
        print(f"Fehler beim Erstellen mit explizitem Schema: {e}")
        print("Versuche alternative Methode...")
        
        # Alternative: Erstelle DataFrame ohne Schema und konvertiere dann
        df_temp = spark.createDataFrame(converted_items)
        print("Temporäres DataFrame Schema:")
        df_temp.printSchema()
        
        # Konvertiere manuell zu gewünschtem Schema
        transformed_df = apply_schema_transformations(df_temp, spark)
    
    print(f"Schema des resultierenden DataFrames:")
    transformed_df.printSchema()
    
    print(f"Anzahl Datensätze: {transformed_df.count()}")
    
    # Speichere als Parquet
    print(f"Speichere Parquet-Datei: {output_file}")
    
    # Reduziere auf eine einzige Partition für eine einzelne Datei
    single_partition_df = transformed_df.coalesce(1)
    
    # Speichere als Parquet
    single_partition_df.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_file)
    
    print("Konvertierung erfolgreich abgeschlossen!")
    print(f"Output-Struktur:")
    print(f"  📁 {output_file}/")
    print(f"     📄 _SUCCESS")
    print(f"     📄 part-00000-xxxxx.snappy.parquet (Hauptdatei)")
    print(f"\nHinweis: Dies ist normales Parquet-Verhalten. Der Ordner enthält deine Daten.")


def main():
    """
    Hauptfunktion mit Kommandozeilenparameter-Parsing.
    """
    parser = argparse.ArgumentParser(
        description="Konvertiert DynamoDB JSON zu Parquet Format"
    )
    parser.add_argument(
        "input_file",
        help="Pfad zur DynamoDB JSON Eingabedatei"
    )
    parser.add_argument(
        "output_file", 
        help="Pfad zur Parquet Ausgabedatei"
    )
    parser.add_argument(
        "--app-name",
        default="DynamoDB-to-Parquet-Converter",
        help="Name der Spark Application (default: DynamoDB-to-Parquet-Converter)"
    )
    parser.add_argument(
        "--executor-memory",
        default="2g",
        help="Executor Memory für Spark (default: 2g)"
    )
    parser.add_argument(
        "--driver-memory", 
        default="1g",
        help="Driver Memory für Spark (default: 1g)"
    )
    
    args = parser.parse_args()
    
    # Erstelle Spark Session
    print("Initialisiere Spark Session...")
    spark = SparkSession.builder \
        .appName(args.app_name) \
        .config("spark.executor.memory", args.executor_memory) \
        .config("spark.driver.memory", args.driver_memory) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    try:
        process_file(spark, args.input_file, args.output_file)
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()