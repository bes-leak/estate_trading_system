package main

import (
	"fmt"

	"github.com/joho/godotenv"

	"github.com/bes-leak/estate_trading_system/backend/internal/estate_filters"
	"github.com/bes-leak/estate_trading_system/backend/internal/service"
)

func main() {
	_ = godotenv.Load()

	priceMin := int64(1_000_000)
	priceMax := int64(8_000_000)
	areaMin := float32(45.5)

	req := service.SearchRequest{
		PriceMin: &priceMin,
		PriceMax: &priceMax,

		Rooms: []int16{1, 2},

		AreaMin: &areaMin,
		AreaMax: nil,

		BBox: nil,

		MetroWithin:  nil,
		ExcludeFirst: true,
	}

	criterion := service.BuildCriterion(req)
	args := &estate_filters.Args{}

	fmt.Println("WHERE:", criterion.SQL(args))

}
