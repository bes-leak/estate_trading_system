package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/bes-leak/estate_trading_system/backend/internal/adapter/geocoder"
	"github.com/bes-leak/estate_trading_system/backend/internal/domain"
)

func main() {
	address := "Государственный университет Дубна"

	var geocoder domain.Geocoder = geocoder.NewNominatim()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	coord, err := geocoder.Geocode(ctx, address)

	if errors.Is(err, domain.ErrAddressNotFound) {
		fmt.Println("address does not found")
		os.Exit(1)
	}
	if err != nil {
		fmt.Println("error:", err)
		os.Exit(1)
	}

	fmt.Printf("%s -> %.6f, %.6f\n", address, coord.Lat, coord.Lon)
}
