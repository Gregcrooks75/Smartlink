import pyodbc
import pandas as pd

def get_connection(server: str, database: str):
    conn_str = (
        "Driver={SQL Server};"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=yes;"
    )
    return pyodbc.connect(conn_str)

def run_query_trusted(query: str, server: str, database: str) -> pd.DataFrame:
    with get_connection(server, database) as conn:
        return pd.read_sql_query(query, conn)
def run_query_with_temp_trusted(query_setup: str, query_final: str, server: str, database: str) -> pd.DataFrame:
    conn = get_connection(server, database)
    cur = conn.cursor()
    
    # Run setup query (e.g., create temp tables)
    cur.execute("SET NOCOUNT ON; " + query_setup)
    while cur.nextset():
        pass  # Advance through all intermediate results

    # Run final query and load results into DataFrame
    df = pd.read_sql_query(query_final, conn)

    # Clean up
    cur.close()
    conn.close()
    
    return df