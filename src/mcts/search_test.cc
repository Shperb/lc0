/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>

#include <iostream>
#include "src/mcts/node.h"

namespace lczero {

// https://github.com/LeelaChessZero/lc0/issues/209
TEST(Node, TestTrim) {
  Node n(NULL, 0,0);
  std::vector<std::pair<float, float>> toTrim = {
      {1.0, 0.1}, {1.5, 0.2}, {3.7, 0.5}, {4.0, 0.55}, {8.1, 1.0}};
  std::vector<std::pair<float, float>> expectedResult = {
      {1.0, 0.2}, {3.7, 0.55}, {8.1, 1.0}};
  n.trimCDF(toTrim, 3);
  EXPECT_EQ(toTrim.size(), expectedResult.size());
  for (int i = 0; i < toTrim.size(); i++) {
    EXPECT_EQ(toTrim[i].first, expectedResult[i].first);
    EXPECT_EQ(toTrim[i].second, expectedResult[i].second);
  }
}

TEST(Node,TestTrimMorePointsThanData) {
  Node n(NULL, 0,0);
  std::vector<std::pair<float, float>> toTrim = {
      {1.0, 0.1}, {1.5, 0.2}, {3.7, 0.5}, {4.0, 0.55}, {8.1, 1.0}};
  std::vector<std::pair<float, float>> expectedResult = {
      {1.0, 0.1}, {1.5, 0.2}, {3.7, 0.5}, {4.0, 0.55}, {8.1, 1.0}};
  n.trimCDF(toTrim, 10);
  EXPECT_EQ(toTrim.size(), expectedResult.size());
  for (int i = 0; i < toTrim.size(); i++) {
    EXPECT_EQ(toTrim[i].first, expectedResult[i].first);
    EXPECT_EQ(toTrim[i].second, expectedResult[i].second);
  }
}

TEST(Node, TestAddCDFMAX) {
  Node n(NULL, 0, 0);
  std::vector<std::pair<float, float>> first = {
      {0.2, 0.5}, {0.7, 1.0}};
  n.discretizationCDF = first;
  //std::vector<std::pair<float, float>> second = {
  //    {1.0, 1.0}};

  std::vector<std::pair<float, float>> expectedResult = { {-1.0, 1.0}};
  Node n1(NULL, 0, 0);
  n1.MakeTerminal(GameResult::WHITE_WON); 
  std::vector<std::pair<float, float>> second = n1.discretizationCDF;

  n.addCDF(second, true);
  EXPECT_EQ(n.discretizationCDF.size(), expectedResult.size());
  for (int i = 0; i < n.discretizationCDF.size(); i++) {
    EXPECT_EQ(n.discretizationCDF[i].first, expectedResult[i].first);
    EXPECT_EQ(n.discretizationCDF[i].second, expectedResult[i].second);
  }
}

TEST(Node, TestAddCDFMAX2) {
  Node n(NULL, 0, 0);
  std::vector<std::pair<float, float>> first = {{0.2, 0.5}, {0.7, 1.0}};
  n.discretizationCDF = first;
  //std::vector<std::pair<float, float>> second = {{-1.0, 1.0}};
  Node n1(NULL, 0, 0);
  n1.MakeTerminal(GameResult::BLACK_WON);
  std::vector<std::pair<float, float>> second = n1.discretizationCDF;

  std::vector<std::pair<float, float>> expectedResult = {{0.2, 0.5},
                                                         {0.7, 1.0}};

  n.addCDF(second, true);
  EXPECT_EQ(n.discretizationCDF.size(), expectedResult.size());
  for (int i = 0; i < n.discretizationCDF.size(); i++) {
    EXPECT_EQ(n.discretizationCDF[i].first, expectedResult[i].first);
    EXPECT_EQ(n.discretizationCDF[i].second, expectedResult[i].second);
  }
}
/*
TEST(Node,TestAddCDFMAX) {
  Node n(NULL, 0,0);
  std::vector<std::pair<float, float>> first = {
      {1.0, 0.25}, {3.0, 0.5}, {4.0, 1.0}};
  n.discretizationCDF = first;
  std::vector<std::pair<float, float>> second = {
      {0.5, 0.2}, {2.0, 0.4}, {5.0, 1.0}};

  std::vector<std::pair<float, float>> expectedResult = {
      {1.0, 0.1}, {3.0, 0.4}, {5.0, 1.0}};

  n.addCDF(second,false);
  EXPECT_EQ(n.discretizationCDF.size(), expectedResult.size());
  for (int i = 0; i < n.discretizationCDF.size(); i++) {
    EXPECT_EQ(n.discretizationCDF[i].first, expectedResult[i].first);
    EXPECT_EQ(n.discretizationCDF[i].second, expectedResult[i].second);
  }
}

TEST(Node,TestAddCDFMIN) {
  Node n(NULL, 0,1);
  std::vector<std::pair<float, float>> first = {
      {1.0, 0.25}, {3.0, 0.5}, {4.0, 1.0}};
  n.discretizationCDF = first;
  std::vector<std::pair<float, float>> second = {
      {0.5, 0.2}, {2.0, 0.4}, {5.0, 1.0}};

  std::vector<std::pair<float, float>> expectedResult = {
      {0.5, 0.4}, {2.0, 0.7}, {4.0, 1.0}};
  n.MakeTerminal(GameResult::BLACK_WON);
  n.addCDF(second,false);
  EXPECT_EQ(n.discretizationCDF.size(), expectedResult.size());
  for (int i = 0; i < n.discretizationCDF.size(); i++) {
    EXPECT_EQ(n.discretizationCDF[i].first, expectedResult[i].first);
    ASSERT_TRUE(abs(n.discretizationCDF[i].second - expectedResult[i].second) < 0.0001);
  }
}
*/

TEST(Node, TestTruncatedNormal) {
  double mu = 0.5;
  int points = 10;
  double sigma = 0.1;
  Node n(NULL, 0,0);
  double step = 1.0 / points;
  for (int i = 0; i < points; i++) {
    double epsilon = 0;
    if (i == points - 1) {
      epsilon = 0.01;
	}
    n.discretizationCDF.push_back(std::make_pair(
            n.NormalTrunkedCDFInverse(step * (i + 1) - epsilon, mu, sigma),
            step * (i + 1)));
  }

  ASSERT_TRUE(true);
}

TEST(Node, TestTruncatedNormalCloseToOne) {
  double mu = 0.99;
  int points = 10;
  double sigma = 0.1;
  Node n(NULL, 0,0);
  double step = 1.0 / points;
  for (int i = 0; i < points; i++) {
    double epsilon = 0;
    if (i == points - 1) {
      epsilon = 0.01;
    }
    n.discretizationCDF.push_back(std::make_pair(
        n.NormalTrunkedCDFInverse(step * (i + 1) - epsilon, mu, sigma),
        step * (i + 1)));
  }

  ASSERT_TRUE(true);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
