package postgres

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/bes-leak/estate_trading_system/backend/internal/domain"
	"github.com/bes-leak/estate_trading_system/backend/internal/estate_filters"
)

type ListingRepo struct {
	pool *pgxpool.Pool
}

func NewListingRepo(pool *pgxpool.Pool) *ListingRepo {
	return &ListingRepo{pool: pool}
}

func (r *ListingRepo) Find(ctx context.Context, criterion estate_filters.Criterion, page domain.Page) ([]domain.Listing, error) {
	query, values := buildQuery(criterion, page)

	rows, err := r.pool.Query(ctx, query, values...)
	if err != nil {
		return nil, fmt.Errorf("find listings: %w", err)
	}
	defer rows.Close()

	listings := make([]domain.Listing, 0, page.Limit)

	for rows.Next() {
		var listing domain.Listing

		err := rows.Scan(&listing.ID, &listing.Price, &listing.Area, &listing.Rooms,
			&listing.Level, &listing.Levels, &listing.Coord.Lat, &listing.Coord.Lon)
		if err != nil {
			return nil, fmt.Errorf("scan listing: %w", err)
		}

		listings = append(listings, listing)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read listings: %w", err)
	}

	return listings, nil
}

func buildQuery(criterion estate_filters.Criterion, page domain.Page) (string, []any) {
	args := &estate_filters.Args{}

	where := criterion.SQL(args)

	limit := args.Add(page.Limit)
	offset := args.Add(page.Offset)

	query := fmt.Sprintf(`
		SELECT l.id, l.price, l.area, l.rooms, l.level, l.levels, b.geo_lat, b.geo_lon
		FROM listings l
		JOIN buildings b ON b.id = l.building_id
		WHERE %s
		ORDER BY l.id
		LIMIT %s OFFSET %s`, where, limit, offset)

	return query, args.Values()
}
