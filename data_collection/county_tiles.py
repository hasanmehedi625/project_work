import geopandas as gpd
import pandas as pd

def load_tile_grid(tile_csv_path):
    from shapely.geometry import box
    tiles = pd.read_csv(tile_csv_path)
    tiles["geometry"] = tiles.apply(
        lambda r: box(r["West"], r["South"], r["East"], r["North"]), axis=1
    )
    return gpd.GeoDataFrame(tiles, geometry="geometry", crs="EPSG:4326")

def get_tiles_for_county(county_name, state_fips, counties_shp_path, tile_csv_path):
    counties = gpd.read_file(counties_shp_path).to_crs("EPSG:4326")
    county = counties[
        (counties["NAME"]    == county_name) &
        (counties["STATEFP"] == str(state_fips).zfill(2))
    ]
    if county.empty:
        raise ValueError(f"County '{county_name}' (STATEFP={str(state_fips).zfill(2)}) not found.")
    tile_gdf = load_tile_grid(tile_csv_path)
    county_geom = county.geometry.union_all()
    matching = tile_gdf[tile_gdf.intersects(county_geom)][
        ["Tile", "h", "v", "North", "South", "West", "East", "geometry"]
    ].reset_index(drop=True)
    return matching


if __name__ == "__main__":
    COUNTIES_SHP = r"C:\Users\mehed\Documents\Pilot Work\tl_2025_us_county\tl_2025_us_county.shp"
    TILE_CSV     = r"C:\WSU Journey\Weekly Report to Prof. Lee\PythonProject\pilot_work\viirs_tiles.csv"

    result = get_tiles_for_county(
        county_name       = "Miami-Dade",
        state_fips        = "12",
        counties_shp_path = COUNTIES_SHP,
        tile_csv_path     = TILE_CSV,
    )
    print(f"Tiles covering Miami-Dade ({len(result)} found):\n")
    print(result[["Tile", "h", "v", "North", "South", "West", "East"]].to_string(index=False))