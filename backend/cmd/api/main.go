package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/joho/godotenv"

	"github.com/bes-leak/estate_trading_system/backend/internal/adapter/geocoder"
	"github.com/bes-leak/estate_trading_system/backend/internal/domain"
	"github.com/bes-leak/estate_trading_system/backend/internal/repository/postgres"
)

func main() {
	// .env is optional: in docker the variables come from the environment
	_ = godotenv.Load()

	address := "Московский государственный университет"

	var coder domain.Geocoder = geocoder.NewNominatim()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	coord, err := coder.Geocode(ctx, address)
	if errors.Is(err, domain.ErrAddressNotFound) {
		log.Fatal("address not found")
	}
	if err != nil {
		log.Fatal("geocode: ", err)
	}

	fmt.Printf("%s -> %.6f, %.6f\n", address, coord.Lat, coord.Lon)

	pool, err := pgxpool.New(ctx, os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatal("connect to database: ", err)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		log.Fatal("database unreachable: ", err)
	}

	var buildings domain.BuildingRepository = postgres.NewBuildingRepo(pool)

	building, err := buildings.EnsureBuilding(ctx, coord)
	if err != nil {
		log.Fatal("ensure building: ", err)
	}

	fmt.Printf("building id=%d features_ready=%t\n", building.ID, building.FeaturesReady)
}
