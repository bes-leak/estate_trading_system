package estate_filters

import "strings"

type Or struct {
	Children []Criterion
}

func (c Or) SQL(args *Args) string {
	if len(c.Children) == 0 {
		return "FALSE"
	}

	parts := make([]string, 0, len(c.Children))

	for _, child := range c.Children {
		parts = append(parts, child.SQL(args))
	}

	return "(" + strings.Join(parts, " OR ") + ")"
}
