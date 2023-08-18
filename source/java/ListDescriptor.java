/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.tika.parser.microsoft.rtf;

/**
 * Contains the information for a single list in the list or list override tables.
 */
public class ListDescriptor {
    public final static int NUMBER_TYPE_BULLET = 23;

    public int id;
    // We record this but don't make use if it today:
    public int templateID;
    // We record this but don't make use if it today:
    public boolean isStyle;
    public int[] numberType = new int[9];

    public boolean isUnordered(int level) {
        return numberType[level] == NUMBER_TYPE_BULLET;
    }
}
